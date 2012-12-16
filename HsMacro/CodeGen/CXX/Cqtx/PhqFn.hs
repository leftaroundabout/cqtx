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

{-# LANGUAGE Rank2Types            #-}
{-# LANGUAGE MultiParamTypeClasses #-}
-- {-# LANGUAGE FunctionalDependencies#-}
{-# LANGUAGE FlexibleInstances     #-}
-- {-# LANGUAGE UndecidableInstances  #-}
{-# LANGUAGE ScopedTypeVariables   #-}

module CodeGen.CXX.Cqtx.PhqFn where

import Control.Monad
import Control.Monad.Writer
import Control.Monad.Reader

import Data.Function
import Data.List
import Data.Monoid
import Data.Ratio
import Data.Maybe
import Data.Tuple
-- import Data.Hashable
-- import Data.HashMap





phqFn :: forall paramLabelsList paramValsList
 . EquilenLists paramLabelsList paramValsList
   => String                           -- ^ Name of the resulting phqFn C++ object
    -> paramLabelsList String          -- ^ Default labels of the parameters
    -> (forall x. PhqfnDefining x
             => paramValsList x -> x)  -- ^ Function definition, as a lambda
    -> CqtxCode()                      -- ^ C++ class and object code for a cqtx fittable physical function corresponding to the given definition
phqFn fnName defaultLabels function = ReaderT $ codeWith where
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
         fnResultTerm :: PhqFuncTerm
         fnResultTerm = function $
               fmap (\i->PhqFnParameter $ PhqIdf i) parameterIds
         fnDimTrace :: DimTracer
         fnDimTrace = function $
               fmap (\i->VardimVar $ PhqIdf i) parameterIds
               
         functionEval :: CXXCode()
         functionEval = do
            cxxLine     $ "auto operator()(const measure& parameters)const -> physquantity {"
            cxxIndent 2 $ do
               result <- cqtxTermImplementation "parameters" fnResultTerm
               cxxLine     $ "return ("++result++");"
            cxxLine     $ "}"
         
         paramExamples :: [(Int,CXXExpression)] -> CXXCode()
         paramExamples helpers = do
            cxxLine     $ "auto example_parameterset( const measure& constraints\\n\
                          \                         , const physquantity& desiredret) -> measure {"
            cxxIndent 2 $ do
               cxxLine     $ "measure example;"
               forM_ helpers $ \(i,helper) -> do
                  cxxLine     $ "example.let(argdrfs["++show i++"]) = "
                                     ++helper++"(constraints, desiredret);"
               cxxLine     $ "return example;"
            cxxLine     $ "}"
         
         dimFetchDeclares :: CXXCode [(Int,CXXExpression)]
         dimFetchDeclares = forM parameterIdsList $ \i -> do
            let functionName = "example_parameter"++show i++"value"
            cxxLine     $ "auto "++functionName++"( const measure& constraints\\n\
                          \                       , const physquantity& desiredret) -> physquantity {"
            cxxIndent 2 $ do
               forM_ (dimExpressionsFor (PhqIdf i) fnDimTrace) $ \decomp -> do
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
                        cxxLine     $ "return ("++result++")"
                     cxxLine     $ "}"
                  cxxLine     $ "}"
               cxxLine     $ "std::cerr << \"Insufficient constraints given to determine the physical dimension\\n\
                                            \  of parameter \\\"\" << *argdrfs["++show i++"] << \"\\\"\
                                             \ in phqfn '"++fnName++"'.\\n Sufficient constraint choices would be:\\n\";"
               forM_ (dimExpressionsFor (PhqIdf i) fnDimTrace) $ \(DimExpression decomp) -> do
                  cxxLine  $ "std::cerr<<\"  \"" ++ concat (intersperse"<<','<<"$
                                [ case fixv of
                                   PhqDimlessConst          -> ""
                                   PhqFnParamVal (PhqIdf j) -> "*argdrfs["++show j++"]"
                                   PhqFnResultVal           -> "\"<function result>\""
                                | (fixv,_) <- decomp ]) ++ " << std::endl;"
               cxxLine     $ "abort();"
            cxxLine     $ "}"
            return (i,functionName)
         
            
         className = fnName++"Function"
         
         constructor = do
            cxxLine     $ className ++"() {"
            cxxIndent 2 $ do
               cxxLine     $ "argdrfs.resize("++show nParams++");"
               forM_ idxedDefaultLabels $ \(n,label) ->
                  cxxLine  $ "argdrfs["++show n++"] = "++show label++";"
            cxxLine     $ "}"
         
         parameterIdsList = map snd $ perfectZip defaultLabels parameterIds 
         
         parameterIds :: paramValsList Int
         parameterIds = buildEquilenList defaultLabels (succ) 0
         idxedDefaultLabels = map swap $ perfectZip defaultLabels parameterIds 
         nParams = length idxedDefaultLabels






data PhqIdf = PhqIdf Int deriving (Eq, Ord, Show)

argderefv :: CXXExpression -> PhqIdf -> String
argderefv paramSource (PhqIdf n) = "argdrfs["++show n++"]("++paramSource++")"

data DimTracer = DimlessConstant
               | VardimVar PhqIdf
               | DimEqualAnd (DimTracer,DimTracer) DimTracer
               | DimtraceProduct [DimTracer]
               | DimtracePower DimTracer Rational
               deriving(Show)


beDimLess :: DimTracer -> DimTracer
beDimLess a = DimEqualAnd (a, DimlessConstant) DimlessConstant


instance Num DimTracer where
  fromInteger _ = DimlessConstant
  
  a + b = (a, b) `DimEqualAnd` a
  
  a - b = (a, b) `DimEqualAnd` a
  
  DimtraceProduct l * tr = DimtraceProduct (tr:l)
  tr * DimtraceProduct l = DimtraceProduct (tr:l)
  a * b = DimtraceProduct [a, b]
  
  negate = id
  
  abs = id
  
  signum a = (a, a) `DimEqualAnd` DimlessConstant

  
instance Fractional DimTracer where
  fromRational _ = DimlessConstant
  
  DimtraceProduct l / tr = DimtraceProduct (recip tr : l)
  a / b = DimtraceProduct [a, recip b]
  
  recip a = DimtracePower a (-1)


instance Floating DimTracer where
  pi = DimlessConstant
  exp = beDimLess; log = beDimLess
  sqrt a = DimtracePower a (1/2)
  a**b = (a,DimlessConstant) `DimEqualAnd` beDimLess b
  a`logBase`b = (a,DimlessConstant) `DimEqualAnd` beDimLess b
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

dimExpNormalForm :: DimExpression -> DimExpression    -- DimExpression must always be normalised
dimExpNormalForm (DimExpression l) = DimExpression . reduce . sortf $ l
 where sortf = sortBy (compare`on`fst)
       reduce ((PhqDimlessConst,_):l) = reduce l
       reduce ((a,r):β@(b,s):l)
        | a==b       = reduce $ (a,r+s):l
        | otherwise  = (a,r) : reduce (β:l)
       reduce l = l

instance Monoid DimExpression where
  mempty = DimExpression[]
  mappend (DimExpression a) (DimExpression b) = dimExpNormalForm $ DimExpression(a++b)
  mconcat l = dimExpNormalForm . DimExpression $ l >>= fixValDecomposition


traceAsValue :: DimTracer -> DimExpression
traceAsValue DimlessConstant = mempty
traceAsValue (VardimVar a) = primDimExpr $ PhqFnParamVal a
traceAsValue (DimEqualAnd (_,_) v) = traceAsValue v
traceAsValue (DimtraceProduct l) = mconcat $ map traceAsValue l
traceAsValue (DimtracePower a q) = expExpMap (q*) $ traceAsValue a

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
dimExpressionsFor idf = go $ primDimExpr PhqFnResultVal
 where go :: DimExpression -> DimTracer -> [DimExpression]
       go rev (DimEqualAnd (ida,idb) also)
        = nubDimExprs $ go rev also ++ go (traceAsValue ida) idb ++ go (traceAsValue idb) ida
       go rev (DimtraceProduct l)
        = nubDimExprs $ l >>= \way -> go (rev<>invprod<>traceAsValue way) way
            where invprod = invDimExp . mconcat $ map traceAsValue l
       go rev (DimtracePower p q) = go (expExpMap(/q)rev) p
       go rev rest
        | (q,p) <- extractFixValExp (PhqFnParamVal idf) $ traceAsValue rest //- rev
        , q /= 0       = [expExpMap(/(-q))p]
        | otherwise    = []







type CXXExpression = String

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
              
              inlineIn e@(PhqFnTempRef n')
               | n'`elem`doomed = inlineIn $ l!!n'
              inlineIn (PhqFnFuncApply f e) = PhqFnFuncApply f $ inlineIn e
              inlineIn (PhqFnInfixApply f a b) = PhqFnInfixApply f (inlineIn a) (inlineIn b)
              inlineIn e = e
              
              
              doomed = filter ((<=1) . length . referingTo) $ map fst indexed
              
              referingTo n = filter (refersTo n) l
              refersTo n (PhqFnTempRef n') = n==n'
              refersTo n (PhqFnFuncApply f e) = refersTo n e
              refersTo n (PhqFnInfixApply f a b) = refersTo n a || refersTo n b
              refersTo _ _ = False

 
 
calculateDimExpression :: DimExpression -> PhqFuncTerm
calculateDimExpression (DimExpression decomp) = product $ map phqfImplement decomp
 where phqfImplement (e, x) = implement e ** fromRational x
       implement (PhqFnParamVal pr) = PhqFnParameter pr
       implement PhqFnResultVal = PhqFnPhysicalConst "desiredret"






newtype LinesBuildup = LinesBuildup {builtupLines :: [String]->[String]}
instance Monoid LinesBuildup where
  mempty = LinesBuildup id
  mappend (LinesBuildup a) (LinesBuildup b) = LinesBuildup (a.b)

linesBuildMap :: (String->String) -> LinesBuildup -> LinesBuildup
linesBuildMap f (LinesBuildup a) = LinesBuildup (map f (a[]) ++)
  
type CXXCode = Writer LinesBuildup

instance Show (CXXCode()) where
  show = unlines . ("do":) . map printout . ($[]) . builtupLines . execWriter
   where printout l = "   cxxLine "++show l

cxxLine :: String -> CXXCode()
cxxLine s = tell $ LinesBuildup (s:)

cxxIndent :: Int -> CXXCode a -> CXXCode a
cxxIndent n = censor $ linesBuildMap (replicate n ' '++)


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





class (Floating a) => PhqfnDefining a

instance PhqfnDefining PhqFuncTerm
instance PhqfnDefining DimTracer





class (Functor l1, Functor l2) => EquilenLists l1 l2 where
  perfectZip :: l1 a -> l2 b -> [(a,b)]
  buildEquilenList :: l1 a -> (b->b) -> b -> l2 b
--   singleList :: l1 a -> [a]
--   unfoldToMatch :: l1 a -> (b -> (c,b)) -> b -> l2 c

data EquilenEnd a = P deriving (Show)
infixr 5 :.
data EquilenCons l a = a :. l a deriving (Show)

instance Functor EquilenEnd where
  fmap _ P = P
instance (Functor l) => Functor (EquilenCons l) where
  fmap f (x:.xs) = f x :. fmap f xs

instance EquilenLists EquilenEnd EquilenEnd where
  perfectZip P P = []
  buildEquilenList P _ _ = P
--   singleList P = []
--   unfoldToMatch P _ _ = P
  
instance (EquilenLists l1 l2) => EquilenLists (EquilenCons l1) (EquilenCons l2) where
  perfectZip (x:.xs) (y:.ys) = (x,y) : perfectZip xs ys
  buildEquilenList (_:.xs) f s = s :. buildEquilenList xs f (f s)
--   singleList (x:.xs) = x:xs
--   unfoldToMatch (_:.xs) uff s = let (y,s') = uff s
--                                 in  y :. unfoldToMatch xs uff s'

instance EquilenLists [] [] where   -- obviously unsafe
  perfectZip = zip
  buildEquilenList [] _ _ = []
  buildEquilenList (_:xs) f s = s : buildEquilenList xs f (f s)
--   singleList = id
--   unfoldToMatch [] _ _ = []
--   unfoldToMatch (_:xs) uff s = let (y,s') = uff s
--                                in  y : unfoldToMatch xs uff s'


-- buildEquilenList l1 f = unfoldToMatch l1 $ (\b->(b,b)) . f
