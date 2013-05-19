--   Copyright 2012 Justus Sagemüller.
-- 
--   This file is part of the Cqtx library.
--    This library is free software: you can redistribute it and/or modify
--   it under the terms of the GNU General Public License as published by
--   the Free Software Foundation, either version 3 of the License, or
--   (at your option) any later version.
--    This library is distributed in the hope that it will be useful,
--   but WITHOUT ANY WARRANTY; without even the implied warranty of
--   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
--   GNU General Public License for more details.
--   You should have received a copy of the GNU General Public License
--   along with this library.  If not, see <http://www.gnu.org/licenses/>.

{-# LANGUAGE PatternGuards         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE Rank2Types            #-}
{-# LANGUAGE ImpredicativeTypes    #-}
{-# LANGUAGE MultiParamTypeClasses #-}
-- {-# LANGUAGE FunctionalDependencies#-}
{-# LANGUAGE FlexibleInstances     #-}
-- {-# LANGUAGE UndecidableInstances  #-}
{-# LANGUAGE ScopedTypeVariables   #-}

-- |
-- Module      : CodeGen.CXX.Cqtx.PhqFn 
-- Copyright   : (c) Justus Sagemüller 2012
-- License     : GPL v3
-- 
-- Maintainer  : (@) sagemuej $ smail.uni-koeln.de
-- Stability   : experimental
-- Portability : portable
-- 
-- While the cqtx library provides quite versatile and /in principle/ very
-- low-requirement fitting of general physical functions (namely without the
-- need for any gradients etc.), in practise this is hampered by the cumbersomeness
-- of defining the fittable_phmsqfn subclasses. The macros in @fitfnmacros.h@ can
-- only partially alleviate this: due to the limits of the C preprocessor, most of
-- the calculation load needs to be done in also cumbersome C++ templates or,
-- more likely, at runtime. At the moment, this is implemented rather poorly, in
-- particular the almost bogosort-quality dimensional analysis.
-- 
-- This Haskell module does not claim to be an optimal solution, but it produces
-- significantly better code than the CPP macros, with very safe and easy invocation.
-- In the future this might become a lot more powerful, since it is also possible
-- to automatically create highly optimised specialised versions of the functions, e.g.
-- squaredistance-calculation on CUDA.
-- 

module CodeGen.CXX.Cqtx.PhqFn( phqFn
                                -- * General Cqtx code generation
                             , module CodeGen.CXX.Code
                             , CqtxCode
                             , CqtxConfig, withDefaultCqtxConfig
                             , PhqfnDefining(..)
                                -- * Length-indexed lists
                             , IsolenEnd(P), IsolenCons(..)
                             ) where
import CodeGen.CXX.Code

import Control.Monad hiding (forM_)
import Control.Monad.Writer hiding (forM_)
import Control.Monad.Reader hiding (forM_)

import Data.Function
import Data.List (sort, sortBy, intersperse, group, find)
import Data.Monoid
import Data.Ratio
import Data.Maybe
import Data.Tuple
import Data.Foldable
import Prelude hiding (foldr, concat, sum, product, any, elem)
-- import Data.Hashable
-- import Data.HashMap






-- | Create a physical function that can be fitted to measured data using the cqtx
-- algorithms such as @evolution_minimizer@. The invocation is similar to the
-- primitive CPP macros, but type-safe and allows full Haskell syntax in the
-- definition – though this can of course be exploited only so much, since phqfns
-- are very limited in their abilities. To make sure these limits are maintained
-- we use universally-quantised arguments (which also suits the implementation very
-- well).
-- 
-- For example, the standard gaussian peak @A⋅exp(-(x-x₀)²/(2⋅σ²))@ could be defined thus:
-- 
-- >  phqFn "gaussPeak" ("x":."x_0":."\\sigma":."A":.P)
-- >                 (\ ( x :.  x0 :.  sigma  :. a :.P)
-- >                   -> let q = (x-x0)/sigma in  a * exp(-0.5 * q^2) )
-- 
-- The use of the type-determined–length lists makes it impossible to accidentally
-- give different numbers of parameter bindings and -labels.
-- 
-- Avoidance of duplicate calculation, as well as rudimentary optimisation, is taken
-- care for by this preprocessor.
phqFn :: forall paramsList . IsolenList paramsList
   => String                           -- ^ Name of the resulting phqFn C++ object
    -> paramsList String          -- ^ Default labels of the parameters
    -> (forall x. PhqfnDefining x
             => paramsList x -> x)  -- ^ Function definition, as a lambda
    -> CqtxCode()                      -- ^ C++ class and object code for a cqtx fittable physical function corresponding to the given definition.
phqFn fnName sclLabels function
  = phqFlatMultiIdFn fnName sclLabels P $ \P -> (P, \scl P -> function scl)


phqFlatMultiIdFn :: forall scalarPrmsList indexerList indexedPrmsList
 . (IsolenList scalarPrmsList, IsolenList indexerList, IsolenList indexedPrmsList)
   => String
    -> scalarPrmsList String
    -> indexerList (String, Maybe Int)
    -> ( indexerList PhqVarIndexer ->
          ( indexedPrmsList (String, PhqVarIndexer)
          , forall x. PhqfnDefining x
             => scalarPrmsList x -> indexedPrmsList (IdxablePhqDefVar x) -> x) )
    -> CqtxCode()
phqFlatMultiIdFn fnName' sclLabels ixerLabels indexedFn = ReaderT $ codeWith where
 codeWith config = do
     cxxLine     $ "                                                                COPYABLE_PDERIVED_CLASS(/*"
     cxxLine     $ "class*/"++className++",/*: public */fittable_phmsqfn) {"
     helpers <- cxxIndent 2 $ dimFetchDeclares
     cxxLine     $ " public:"
     cxxIndent 2 $ do constructor
                      paramExamples helpers
                      functionEval
     cxxLine     $ "} " ++ fnName ++ ";"
   where 
         function :: forall x . PhqfnDefining x 
                      => scalarPrmsList x -> indexedPrmsList (IdxablePhqDefVar x) -> x
         ixps :: indexedPrmsList (String, PhqVarIndexer)
         
         (ixps, function) = indexedFn indexers
 
         fnResultTerm :: PhqFuncTerm
         fnResultTerm = function (fmap (\i->PhqFnParameter $ PhqIdf i) scalarParamIds)
                                 (perfectZipWith (\vni (vnm, PhqVarIndexer ix inm) 
                                          -> IdxablePhqDefVar $ q vni vnm ix inm)
                                      ixableParamIds ixps)
              where q vni vnm ix inm (PhqVarIndexer ix' inm')
                     | ix'==ix    = PhqFnParameter $ PhqIdf vni
                     | otherwise  = error $ "In HsMacro-defined phqfunction '"++fnName++"':\n"
                                            " Using wrong indexer '"++inm'++"' (not adapted range!) \
                                            \for indexing variable '"++vnm++"'.\n"
                                            ++" Correct indexer would be '"++inm++"'."
         fnDimTrace :: DimTracer
         fnDimTrace = function (fmap (VardimVar . PhqIdf) scalarParamIds)
                               (fmap (VardimVar . PhqIdf) ixableParamIds)
               
         functionEval :: CXXCode()
         functionEval = do
            cxxLine     $ "auto operator()(const measure& parameters)const -> physquantity {"
            cxxIndent 2 $ do
--                cxxLine     $ "std::cout << \"evaluate "++fnName++" with parameters\\n\""
--                cxxLine     $ "          << parameters << std::endl;"
               result <- cqtxTermImplementation "parameters" fnResultTerm
               cxxLine     $ "return ("++result++");"
            cxxLine     $ "}"
         
         paramExamples :: [(Int,CXXExpression)] -> CXXCode()
         paramExamples helpers = do
            cxxLine     $ "auto example_parameterset( const measure& constraints\n\
                          \                         , const physquantity& desiredret)const override -> measure {"
            cxxIndent 2 $ do
               cxxLine     $ "measure example;"
               forM_ helpers $ \(i,helper) -> do
                  cxxLine     $ "if(!constraints.has(*argdrfs["++show i++"]))"
                  cxxLine     $ "  example.let(*argdrfs["++show i++"]) = "
                                     ++helper++"(constraints, desiredret);"
               cxxLine     $ "return example;"
            cxxLine     $ "}"
         
         dimFetchDeclares :: CXXCode [(Int,CXXExpression)]
         dimFetchDeclares = forM (toList scalarParamIds) $ \i -> do
            let functionName = "example_parameter"++show i++"value"
            let resultOptions = relevantDimExprsFor (PhqIdf i) fnDimTrace
            
            cxxLine     $ "auto "++functionName++"( const measure& constraints\n\
                          \                       , const physquantity& desiredret)const -> physquantity {"
            cxxIndent 2 $ do
               forM_ resultOptions $ \decomp -> do
                  cxxLine     $ "{"
                  cxxIndent 2 $ do
                     cxxLine     $ "bool thisdecomp_failed = false;"
                     forM_ (fixValDecomposition decomp) $ \(e,_) -> case e of
                        PhqFnParamVal (PhqIdf n) -> do
                           cxxLine    $ "if(!constraints.has(*argdrfs["++show n++"]))"
                           cxxLine    $ "  thisdecomp_failed = true;"
                        _ -> return ()
                     cxxLine     $ "if(!thisdecomp_failed) {"
                     cxxIndent 2 $ do
                        result <- cqtxTermImplementation "constraints" (calculateDimExpression decomp)
                        cxxLine     $ "physquantity result = ("++result++");"
                        cxxLine     $ "return result.werror(result);"
                     cxxLine     $ "}"
                  cxxLine     $ "}"
               cxxLine     $ "std::cerr << \"Insufficient constraints given to determine the physical dimension\\n\
                                            \  of parameter \\\"\" << *argdrfs["++show i++"] << \"\\\"\
                                             \ in phqfn '"++fnName++"'.\\n Sufficient constraint choices would be:\\n\";"
               forM_ resultOptions $ \(DimExpression decomp) -> do
                  cxxLine  $ "std::cerr<<\"  \"" ++ concat (intersperse"<<','"$
                                [ case fixv of
                                   PhqDimlessConst          -> ""
                                   PhqFnParamVal (PhqIdf j) -> "<<miniTeX(*argdrfs["++show j++"])"
                                   PhqFnResultVal           -> "<<\"<function result>\""
                                | (fixv,_) <- decomp ]) ++ " << std::endl;"
               cxxLine     $ "abort();"
            cxxLine     $ "}"
            return (i,functionName)
         
         rangesDecl :: CXXCode [CXXExpression]
         rangesDecl = do
             let rangeConstsNeeded = filter ((==head indizesPrefix) . head)
                                                    $ toList ixableParamRanges
             cxxLine $ "unsigned "
                        ++ intercalate ", " rangeConstsNeeded ++ ";"
             return rangeConstsNeeded
            
         className = fnName++"Function"
         fnName = makeSafeCXXIdentifier fnName'
         
         offsetMgr :: CXXCode()
         offsetMgr = do
            cxxLine     $ "void manage_offsets() {"
            cxxIndent 2 $ do
               cxxLine     $ "unsigned stackp = "++show(isoLength scalarParamIds)++";"
               forM_ 
            
         
         constructor :: [CXXExpression] -> CXXCode()
         constructor cstrArgs = do
            cxxLine     $ className ++"("++args++")"++initialisers++" {"
            cxxIndent 2 $ do
               cxxLine     $ "manage_offsets();"
               forM_ idxedDefaultLabels $ \(n,label) ->
                  cxxLine  $ "argdrfs["++show n++"] = "++show label++";"
            cxxLine     $ "}"
          where args = intercalate ", " cstrArgs
                initialisers
                  | null cstrArgs  = ""
                  | otherwise      = ": " ++ intercalate ", "
                                      ( map (\a -> a++"("++a++")" ) cstrArgs )
         
         indexers :: indexerList PhqVarIndexer
         indexers = perfectZipWith PhqVarIndexer (enumFrom' 0) ixerLabels
         
         scalarParamIds :: scalarPrmsList Int
         scalarParamIds = enumFrom' 0
         
         ixableParamIds :: indexedPrmsList Int
         ixableParamIds = enumFrom' nParams
         
         ixableParamRanges :: indexedPrmsList CXXExpression
         ixableParamRanges = fmap (rngFind . snd) ixps
          where rngFind (PhqVarIndexer ix _)
                  = case ixerLabels !!@ ix of
                     (_, Just n) = show n
                     (name, _  ) = indizesPrefix ++ makeSafeCXXIdentifier name 
                                         ++ indexRangePostfix
         
         indizesPrefix = "paramindex_"
         indexRangePostfix = "_range"
         indexOffsetPostfix = "_offset"
         
         
         idxedDefaultLabels = perfectZip scalarParamIds sclLabels
         nParams = isoLength scalarParamIds






data PhqIdf = PhqIdf Int deriving (Eq, Ord, Show)

argderefv :: CXXExpression -> PhqIdf -> String
argderefv paramSource (PhqIdf n) = "argdrfs["++show n++"]("++paramSource++")"

data DimTracer = DimlessConstant (Maybe Rational)
               | VardimVar PhqIdf
               | DimEqualities [(DimTracer,DimTracer)] -- pairs of expressions that should have equal phys-dimension, but not necessarily related to the result
                               [DimTracer]             -- expressions that should have the same phys-dimension as the result
               | DimtraceProduct [DimTracer]
               | DimtracePower DimTracer Rational
               deriving(Show)

unknownDimlessConst :: DimTracer
unknownDimlessConst = DimlessConstant Nothing

beDimLess :: DimTracer -> DimTracer
beDimLess a = DimEqualities [(a, unknownDimlessConst)] [unknownDimlessConst]
biBeDimLess :: DimTracer -> DimTracer -> DimTracer
biBeDimLess a b = DimEqualities (map(,unknownDimlessConst)[a,b]) [unknownDimlessConst]


instance Num DimTracer where
  fromInteger = DimlessConstant . Just . fromInteger
  
  a + b = DimEqualities [(a,b)] [a,b]
  
  a - b = DimEqualities [(a,b)] [a,b]
  
  DimtraceProduct l * tr = DimtraceProduct (tr:l)
  tr * DimtraceProduct l = DimtraceProduct (tr:l)
  a * b = DimtraceProduct [a, b]
  
  negate = id
  
  abs = id
  
  signum a = DimEqualities [(a,a)] [unknownDimlessConst]

  
instance Fractional DimTracer where
  fromRational = DimlessConstant . Just
  
  DimtraceProduct l / tr = DimtraceProduct (recip tr : l)
  a / b = DimtraceProduct [a, recip b]
  
  recip a = DimtracePower a (-1)


instance Floating DimTracer where
  pi = unknownDimlessConst
  exp = beDimLess; log = beDimLess
  sqrt a = DimtracePower a (1/2)
  a**(DimlessConstant (Just b)) = DimtracePower a b
  a**b = biBeDimLess a b
  logBase = biBeDimLess
  sin = beDimLess; cos = beDimLess; tan = beDimLess
  asin = beDimLess; acos = beDimLess; atan = beDimLess
  sinh = beDimLess; cosh = beDimLess; tanh = beDimLess
  asinh = beDimLess; acosh = beDimLess; atanh = beDimLess
  


data PhqFixValue = PhqDimlessConst
                 | PhqFnParamVal PhqIdf
                 | PhqFnResultVal
                 deriving(Eq, Ord, Show)

newtype DimExpression = DimExpression { fixValDecomposition :: [(PhqFixValue, Rational)] }
            deriving(Eq, Ord, Show)

expExpMap :: (Rational -> Rational) -> DimExpression -> DimExpression
expExpMap f (DimExpression l) = DimExpression[ (a,f r) | (a,r)<-l ]

invDimExp :: DimExpression -> DimExpression
invDimExp = expExpMap negate

primDimExpr :: PhqFixValue -> DimExpression
primDimExpr v = DimExpression[(v,1)]

dimExpNormalForm :: DimExpression -> DimExpression    -- DimExpression must always be normalised to this form
dimExpNormalForm (DimExpression l) = DimExpression . reduce . sortf $ l
 where sortf = sortBy (compare`on`fst)
       reduce ((PhqDimlessConst,_):l') = reduce l'
       reduce ((a,r):β@(b,s):l')
        | a==b       = reduce $ (a,r+s):l'
        | otherwise  = (a,r) : reduce (β:l')
       reduce l' = l'

instance Monoid DimExpression where
  mempty = DimExpression[]
  mappend (DimExpression a) (DimExpression b) = dimExpNormalForm $ DimExpression(a++b)
  mconcat l = dimExpNormalForm . DimExpression $ l >>= fixValDecomposition

dimExprnComplexity :: DimExpression -> Integer
dimExprnComplexity (DimExpression l) = sum $ map complexity l
 where complexity (_, fr) = abs(denominator fr) + abs(numerator fr - 1)

compareDimsBasis :: DimExpression -> DimExpression -> Ordering
compareDimsBasis (DimExpression l) (DimExpression r)
             = comp (map fst l) (map fst r)
 where comp [] [] = EQ
       comp _ [] = GT
       comp [] _ = LT
       comp (l:ls) (r:rs)
        | l<r   = GT
        | l>r   = LT
        | l==r  = comp ls rs

strictlySimplerDimsBasis :: DimExpression -> DimExpression -> Bool
strictlySimplerDimsBasis (DimExpression l) (DimExpression r)
             = comp (map fst l) (map fst r)
 where comp _ [] = False
       comp [] _ = True
       comp (l:ls) (r:rs)
        | l<r   = False
        | l>r   = comp (l:ls) rs
        | l==r  = comp ls rs


traceAsValue :: DimTracer -> [DimExpression]
traceAsValue (DimlessConstant _) = return mempty
traceAsValue (VardimVar a) = return (primDimExpr $ PhqFnParamVal a)
traceAsValue (DimEqualities _ v) = v >>= traceAsValue
traceAsValue (DimtraceProduct l) = map mconcat . sequence $ map traceAsValue l
traceAsValue (DimtracePower a q) = map (expExpMap (q*)) $ traceAsValue a

(//-) :: DimExpression -> DimExpression -> DimExpression
a //- b = a <> invDimExp b

extractFixValExp :: PhqFixValue -> DimExpression -> (Rational, DimExpression)
extractFixValExp v = (\(a,b)->(a,DimExpression b)) . go id . fixValDecomposition
 where go acc (t@(w,r):l)
        | v==w  = (r, acc l)
        | v>w   = go (acc.(t:)) l
       go acc e = (0, acc e)

nubDimExprs :: [DimExpression] -> [DimExpression]
nubDimExprs = map head . group . sort



dimExpressionsFor :: PhqIdf -> DimTracer -> [DimExpression]
dimExpressionsFor idf e = go e $ primDimExpr PhqFnResultVal
 where go :: DimTracer -> DimExpression -> [DimExpression]
       go (DimEqualities mutualEqs resEqs) rev
        = nubDimExprs $ ((`go`rev) =<< resEqs) ++ ( do
                   (ida,idb) <- mutualEqs
                   (go idb =<< traceAsValue ida)
                                    ++ (go ida =<< traceAsValue idb) )
       go e@(DimtraceProduct l) rev = nubDimExprs $ do
            way <- l
            value <- traceAsValue way
            invprod <- map invDimExp $ traceAsValue e
            go way $ rev<>invprod<>value
       go (DimtracePower p q) rev = go p $ expExpMap(/q) rev
       go rest rev = do
            value <- traceAsValue rest
            let (q,p) = extractFixValExp (PhqFnParamVal idf) $ value //- rev
            guard (q /= 0)
            [expExpMap(/(-q))p]


relevantDimExprsFor :: PhqIdf -> DimTracer -> [DimExpression]
relevantDimExprsFor idf = prune . cplxtySort . dimExpressionsFor idf
 where cplxtySort = sortBy (compare `on` negate . dimExprnComplexity)
       prune [] = []
       prune (e:es)
        | any(`strictlySimplerDimsBasis`e)es  = prune es
        | otherwise                           = e : prune es





newtype CXXFunc = CXXFunc {wrapCXXFunc :: CXXExpression -> CXXExpression}
newtype CXXInfix = CXXInfix {wrapCXXInfix :: CXXExpression -> CXXExpression -> CXXExpression}

instance Eq CXXFunc where
  CXXFunc f==CXXFunc g  = f""==g""
instance Eq CXXInfix where
  CXXInfix f==CXXInfix g  = ""`f`""==""`g`""

cxxFunction :: CXXExpression -> CXXFunc
cxxFunction s = CXXFunc $ \e -> s++"("++e++")"
cxxInfix :: CXXExpression -> CXXInfix
cxxInfix s = CXXInfix $ \e1 e2 -> "("++e1++s++e2++")"



type PhysicalCqtxConst = String

data PhqFuncTerm = PhqFnDimlessConst Double
                 | PhqFnPhysicalConst PhysicalCqtxConst
                 | PhqFnTempRef Int
                 | PhqFnParameter PhqIdf
                 | PhqFnFuncApply CXXFunc PhqFuncTerm
                 | PhqFnInfixApply CXXInfix PhqFuncTerm PhqFuncTerm
                 | PhqFnInfixFoldOverIndexer PhqVarIndexer CXXInfix PhqFuncTerm PhqFuncTerm
                 deriving (Eq)
                 
cxxFuncPhqApply :: CXXExpression -> PhqFuncTerm -> PhqFuncTerm
cxxFuncPhqApply s = PhqFnFuncApply $ cxxFunction s
cxxInfixPhqApply :: CXXExpression -> PhqFuncTerm -> PhqFuncTerm -> PhqFuncTerm
cxxInfixPhqApply s = PhqFnInfixApply $ cxxInfix s

isPrimitive :: PhqFuncTerm -> Bool
isPrimitive (PhqFnDimlessConst _) = True
isPrimitive (PhqFnPhysicalConst _) = True
isPrimitive (PhqFnParameter _) = True
isPrimitive _ = False


instance Num PhqFuncTerm where
  fromInteger = PhqFnDimlessConst . fromInteger
  
  (+) = cxxInfixPhqApply"+"
  (-) = cxxInfixPhqApply"-"
  
  a*(PhqFnDimlessConst 1) = a
  (PhqFnDimlessConst 1)*b = b
  a*b
   | a==b       = PhqFnFuncApply (CXXFunc $ \e -> "("++e++").squared()") a
   | otherwise  = cxxInfixPhqApply"*" a b
  
  negate = cxxFuncPhqApply"-"
  abs = cxxFuncPhqApply"abs"
  signum = cxxFuncPhqApply"sgn"

instance Fractional PhqFuncTerm where
  fromRational = PhqFnDimlessConst . fromRational
  
  (/) = cxxInfixPhqApply"/"
  
  recip = cxxFuncPhqApply"inv"

instance Floating PhqFuncTerm where
  pi = PhqFnDimlessConst pi
  exp = cxxFuncPhqApply"exp"
  log = cxxFuncPhqApply"ln"
  sqrt = cxxFuncPhqApply"sqrt"
  a**(PhqFnDimlessConst 1) = a
  a**(PhqFnDimlessConst x) = PhqFnFuncApply
           (CXXFunc $ \base -> "(("++base++").to("++show x++"))") a
  a**x = exp(log a*x)
  sin = cxxFuncPhqApply"sin"
  cos = cxxFuncPhqApply"cos"
  tan = cxxFuncPhqApply"tan"
  tanh = cxxFuncPhqApply"tanh"
  asin = error "asin of physquantities not currently implemented."
  acos = error "acos of physquantities not currently implemented."
  atan = error "atan of physquantities not currently implemented."
  sinh = error "sinh of physquantities not currently implemented."
  cosh = error "cosh of physquantities not currently implemented."
  asinh = error "asinh of physquantities not currently implemented."
  acosh = error "acosh of physquantities not currently implemented."
  atanh = error "atanh of physquantities not currently implemented."




-- 'seqPrunePhqFuncTerm' could be implemented much more efficiently by replacing
-- the lists with hash tables. This has low priority, since any function complicated
-- enough for this to take noteworthy time would always take yet a lot more time
-- when used with the cqtx algorithms.

-- instance Hashable PhqFuncTerm where
--   hash (PhqFnDimlessConst a) = hash"DimlessConst"`combine`hash a
--   hash (PhqFnPhysicalConst a) = hash"PhysicalConst"`combine`hash a
--   hash (PhqFnParameter a) = hash"Parameter"`combine`hash a
--   hash (PhqFnFuncApply f a) = hash"FuncApply"`combine`hash f`combine`hash a
--   hash (PhqFnInfixApply f a b) = hash"InfixApply"`combine`hash f`combine`hash a`combine`hash b
                  
--                   migrateRs (PhqFnTempRef n) = PhqFnTempRef (n+nLhsRefs)
--                   migrateRs (PhqFnFuncApply f a) = PhqFnFuncApply f (migrateRs a)
--                   migrateRs (PhqFnInfixApply f a b) = PhqFnInfixApply f (migrateRs a) (migrateRs b)
--                   migrateRs c = c



seqPrunePhqFuncTerm :: PhqFuncTerm -> (PhqFuncTerm,[(Int,PhqFuncTerm)])
seqPrunePhqFuncTerm = prune . reverse . go []
 where go :: [PhqFuncTerm] -> PhqFuncTerm -> [PhqFuncTerm]
       go acc term
        | (l, _:r) <- break((==term).expandRefs acc) acc 
                  = refIn acc (length r) : acc
       go acc x@(PhqFnFuncApply f a)
        | isPrimitive a = x:acc
        | otherwise
            = let acc' = go acc a
              in  PhqFnFuncApply f (refIn acc' $ length acc'-1) : acc'
       go acc x@(PhqFnInfixApply f a b)
        | isPrimitive a, isPrimitive b = x:acc
        | isPrimitive a
            = let acc' = go acc b
              in  PhqFnInfixApply f a (refIn acc' $ length acc'-1) : acc'
        | isPrimitive b
            = let acc' = go acc a
              in  PhqFnInfixApply f (refIn acc' $ length acc'-1) b : acc'
        | otherwise
            = let accL = go acc a
                  accR = go accL b
                  [nLhsRefs,nRhsRefs] = map length [accL,accR]
              in  PhqFnInfixApply f (refIn accR $ nLhsRefs-1)
                                    (refIn accR $ nRhsRefs-1) : accR
       go acc term = term : acc
       
       expandRefs :: [PhqFuncTerm] -> PhqFuncTerm -> PhqFuncTerm
       expandRefs rrl c = expnd c
        where reflist = reverse rrl
              expnd (PhqFnTempRef n) = expnd $ reflist!!n
              expnd (PhqFnFuncApply f t) = PhqFnFuncApply f $ expnd t
              expnd (PhqFnInfixApply f a b) = PhqFnInfixApply f (expnd a) (expnd b)
              expnd c = c
       
       refIn rrl n = chase(PhqFnTempRef n)
        where reflist = reverse rrl
              chase (PhqFnTempRef n)
                | r'@(PhqFnTempRef n')<-reflist!!n = chase r'
              chase c = c
       
       prune :: [PhqFuncTerm] -> (PhqFuncTerm, [(Int, PhqFuncTerm)])
       prune l = ( inlineIn $ last l, catMaybes $ map zap indexed )
        where indexed = zip[0..] l
              
              zap (n, e)
               | n`elem`doomed  = Nothing
               | otherwise      = Just (n, inlineIn e)
              
              inlineIn (PhqFnTempRef n')
               | n'`elem`doomed = inlineIn $ l!!n'
              inlineIn (PhqFnFuncApply f e) = PhqFnFuncApply f $ inlineIn e
              inlineIn (PhqFnInfixApply f a b) = PhqFnInfixApply f (inlineIn a) (inlineIn b)
              inlineIn e = e
              
              
              doomed = filter ((<=1) . length . referingTo) $ map fst indexed
              
              referingTo n = filter (refersTo n) l
              refersTo n (PhqFnTempRef n') = n==n'
              refersTo n (PhqFnFuncApply _ e) = refersTo n e
              refersTo n (PhqFnInfixApply _ a b) = refersTo n a || refersTo n b
              refersTo _ _ = False

 
 
calculateDimExpression :: DimExpression -> PhqFuncTerm
calculateDimExpression (DimExpression decomp) = product $ map phqfImplement decomp
 where phqfImplement (e, x) = implement e ** fromRational x
       implement (PhqFnParamVal pr) = PhqFnParameter pr
       implement PhqFnResultVal = PhqFnPhysicalConst "desiredret"
       implement PhqDimlessConst = PhqFnDimlessConst 1







cqtxTermImplementation :: CXXExpression -> PhqFuncTerm -> CXXCode CXXExpression
cqtxTermImplementation paramSource e = do
           forM_ seqChain $ \(rn,se) ->
              cxxLine $ "physquantity tmp"++show rn++" = "++showE se++";"
           return $ showE result
 where (result,seqChain) = seqPrunePhqFuncTerm e
       showE (PhqFnDimlessConst x) = "("++show x++"*real1)"
       showE (PhqFnPhysicalConst x) = "("++x++")"
       showE (PhqFnParameter x) = argderefv paramSource x
       showE (PhqFnTempRef n) = "tmp"++show n
       showE (PhqFnFuncApply (CXXFunc f) a) = f $ showE a
       showE (PhqFnInfixApply (CXXInfix f) a b) = showE a `f` showE b



type CqtxConfig = ()
type CqtxCode = ReaderT CqtxConfig CXXCode

withDefaultCqtxConfig :: CqtxCode a -> CXXCode a
withDefaultCqtxConfig = flip runReaderT ()



newtype IdxablePhqDefVar x
 = IdxablePhqDefVar {
     indexisePhqDefVar :: PhqVarIndexer -> x }

data PhqVarIndexer = PhqVarIndexer 
  { phqVarIndexerId :: Int
  , phqVarIndexerName :: String
  } deriving(Eq)


class (Floating a) => PhqfnDefining a where
  sumOverIdx :: PhqVarIndexer
      -> ((IdxablePhqDefVar a->a) -> a) -> a

instance PhqfnDefining PhqFuncTerm where
  sumOverIdx i summand
      = PhqFnInfixFoldOverIndexer i (cxxInfix"+") 0 
          . forbidDupFold i "sum"
          . summand $ \(IdxablePhqDefVar e) -> e i
         

instance PhqfnDefining DimTracer where
  sumOverIdx i summand = summand $
      \(IdxablePhqDefVar e) -> e i




forbidDupFold :: PhqVarIndexer -> String -> PhqFuncTerm -> PhqFuncTerm
forbidDupFold i@(PhqVarIndexer _ nmm) fnm = go
 where go(PhqFnInfixFoldOverIndexer i' ifx ini tm)
        | i'==i      = error $ "Duplicate "++fnm++" over index "++nmm
        | otherwise  = PhqFnInfixFoldOverIndexer i' ifx (go ini) (go tm)
       go(PhqFnFuncApply f tm) = PhqFnFuncApply f $ go tm
       go(PhqFnInfixApply ifx l r) = PhqFnInfixApply ifx (go l) (go r)
       go e = e




class (Functor l, Foldable l) => IsolenList l where
  perfectZipWith :: (a->b->c) -> l a -> l b -> l c
  buildIsolenList :: (b->b) -> b -> l b
  isoLength :: l a -> Int
  isoLength = length . toList
  enumFrom' :: Enum a => a -> l a
  enumFrom' = buildIsolenList succ
  (!!@) :: Int -> l a -> a

perfectZip :: IsolenList l => l a -> l b -> l (a,b)
perfectZip = perfectZipWith(,)



data IsolenEnd a = P deriving (Show)
infixr 5 :.
data IsolenCons l a = a :. l a deriving (Show)

instance Functor IsolenEnd where
  fmap _ P = P
instance (Functor l) => Functor (IsolenCons l) where
  fmap f (x:.xs) = f x :. fmap f xs

instance Foldable IsolenEnd where
  foldr _ = const
instance (Foldable l) => Foldable (IsolenCons l) where
  foldr f ini (x:.xs) = x `f` foldr f ini xs

instance IsolenList IsolenEnd where
  perfectZipWith _ P P = P
  buildIsolenList _ _ = P
  isoLength _ = 0
  P !!@ (-1) = undefined
  
instance (IsolenList l) => IsolenList (IsolenCons l) where
  perfectZipWith f (x:.xs) (y:.ys) = f x y :. perfectZip xs ys
  buildIsolenList f s = s :. buildIsolenList f (f s)
  isoLength (_:.xs) = 1 + isoLength xs
  (x:._) !!@ 0 = x
  (_:.xs) !!@ n = xs !!@ (n-1)


-- instance IsolenList [] where   -- obviously unsafe
--   perfectZipWith = zipWith
--   buildIsolenList = iterate
--   enumFrom' = enumFrom
--   singleList = id
--   unfoldToMatch [] _ _ = []
--   unfoldToMatch (_:xs) uff s = let (y,s') = uff s
--                                in  y : unfoldToMatch xs uff s'


-- buildIsolenList l1 f = unfoldToMatch l1 $ (\b->(b,b)) . f
