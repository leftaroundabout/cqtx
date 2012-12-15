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

{-# LANGUAGE Rank2Types         #-}

module CodeGen.CXX.Cqtx.PhqFn where

import Data.Ratio
import Data.List
import Data.Function
import Data.Monoid
-- import Data.Hashable
-- import Data.HashMap

type PhqIdf = String

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

-- data DimTheorem = DimEqualTo PhqFixValue
--                 | DimPowerOf DimTheorem Rational
--                 | DimProductOf [DimTheorem]
-- 
-- instance Eq DimTheorem where
--   DimEqualTo a==DimEqualTo b                 = a==b
--   DimPowerOf (DimEqualTo a) r==DimEqualTo s  = r==1 && a==b
--   DimPowerOf a r==DimPowerOf b s             = DimPowerOf a (r/s)==b
--   DimPowerOf (DimPowerOf a s) r==b           = DimPowerOf a (r*s)==b
--   DimPowerOf (DimProductOf l) r==b           = DimProductOf (map (`DimPowerOf`r) l)==b
--   DimProductOf l==b           = DimProductOf (map (`DimPowerOf`r) l)==b

--  eqResult (traceAsValue ida) (traceAsValue idb)
--             where eqResult e1 e2
--                    | (q,res) <- extractFixValExp idf (e1 //- e2) /= 0
--                    , q/=0         = expExpMap (/q) res

-- elemFoci :: [a] -> [(a,[a])]
-- elemFoci = go id
--  where go acc [] = []
--        go acc (e:l) = (e, acc l) : go (acc.(e:)) l



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



type CXXFunc = String
type CXXInfix = String

type PhysicalCqtxConst = String

data PhqFuncTerm = PhqFnDimlessConst Double
                 | PhqFnPhysicalConst PhysicalCqtxConst
                 | PhqFnTempRef Int
                 | PhqFnParameter PhqIdf
                 | PhqFnFuncApply CXXFunc PhqFuncTerm
                 | PhqFnInfixApply CXXInfix PhqFuncTerm PhqFuncTerm
                 deriving (Eq, Show)

isPrimitive :: PhqFuncTerm -> Bool
isPrimitive (PhqFnDimlessConst _) = True
isPrimitive (PhqFnPhysicalConst _) = True
isPrimitive (PhqFnParameter _) = True
isPrimitive _ = False


instance Num PhqFuncTerm where
  fromInteger = PhqFnDimlessConst . fromInteger
  
  (+) = PhqFnInfixApply "+"
  (-) = PhqFnInfixApply "-"
  (*) = PhqFnInfixApply "*"
  
  negate = PhqFnFuncApply "-"
  abs = PhqFnFuncApply "abs"
  signum = PhqFnFuncApply "sgn"

instance Fractional PhqFuncTerm where
  fromRational = PhqFnDimlessConst . fromRational
  
  (/) = PhqFnInfixApply "/"
  
  recip = PhqFnFuncApply "inv"

instance Floating PhqFuncTerm where
  pi = PhqFnDimlessConst pi
  exp = PhqFnFuncApply "exp"
  log = PhqFnFuncApply "ln"
  sqrt = PhqFnFuncApply "sqrt"
  sin = PhqFnFuncApply "sin"
  cos = PhqFnFuncApply "cos"
  tan = PhqFnFuncApply "tan"
  tanh = PhqFnFuncApply "tanh"
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
       
       prune l = (last l, filter(isRefd . fst) $ zip[0..] l)
        where isRefd n = any (refersTo n) l
              refersTo n (PhqFnTempRef n') = n==n'
              refersTo n (PhqFnFuncApply f e) = refersTo n e
              refersTo n (PhqFnInfixApply f a b) = refersTo n a || refersTo n b
              refersTo _ _ = False
--        prune l = (last l, filter(isRefd . fst) $ zip[0..] l)
--         where isRefd n = filter (refersTo n) l
--               refersTo n (PhqFnTempRef n') = n==n'
--               refersTo n (PhqFnFuncApply f e) = refersTo n e
--               refersTo n (PhqFnInfixApply f a b) = refersTo n a || refersTo n b
--               refersTo _ _ = False
              
              
--        dropPrimitives l@(c:r)
--         | isPrimitive c = dropPrimitives r
--        dropPrimitives l = l
       
--        seqdef (expr:l) = defrep expr []
--         where refs = reverse l
--               
--               defrep (PhqFnTempRef n) acc
--               defrep term acc = term:acc


cqtxTermImplementation :: PhqFuncTerm -> [String]
cqtxTermImplementation e
           = [ "physquantity tmp"++show rn++" = "++showE e++";" | (rn,e)<-seqChain ]
           ++ ["return " ++ showE result ++ ";"]
 where (result,seqChain) = seqPrunePhqFuncTerm e
       showE (PhqFnDimlessConst x) = "("++show x++"*real1)"
       showE (PhqFnPhysicalConst x) = show x
       showE (PhqFnParameter x) = show x
       showE (PhqFnTempRef n) = "tmp"++show n
       showE (PhqFnFuncApply f a) = f++"("++showE a++")"
       showE (PhqFnInfixApply f a b) = "("++showE a++f++showE b++")"