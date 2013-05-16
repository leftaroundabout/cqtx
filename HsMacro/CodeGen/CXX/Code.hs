
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

{-# LANGUAGE FlexibleInstances     #-}

-- |
-- Module      : CodeGen.CXX.Code
-- Copyright   : (c) Justus Sagemüller 2012
-- License     : GPL v3
-- 
-- Maintainer  : (@) sagemuej $ smail.uni-koeln.de
-- Stability   : experimental
-- Portability : portable
-- 
-- A writer monad for building up C++ code, to use Haskell as a (way more powerful)
-- replacement for the C preprocessor. Unlike that, it is designed to produce
-- reasonably /human-readable/ code, so the resulting files can also be used
-- as \"proper\" code files, without Haskell available.


module CodeGen.CXX.Code( CXXExpression
                       , CXXCode
                       , cxxLine, cxxIndent
                       , cxxCodeString
                       , makeSafeCXXIdentifier
                       , exportCXXinliningHeader
                       ) where

import Control.Monad
import Control.Monad.Writer
-- import Control.Monad.Reader

import Data.Function
import Data.List
import Data.Monoid
import Data.Char
import Data.Word
-- import Data.Tuple
-- import Data.Hashable
-- import Data.HashMap

import System.IO
import System.Random

import Numeric


type CXXExpression = String


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


cxxCodeString :: CXXCode() -> String
cxxCodeString = showBuildup . execWriter
 where showBuildup (LinesBuildup buf) = unlines $ buf []


makeSafeCXXIdentifier :: CXXExpression -> CXXExpression
makeSafeCXXIdentifier = killDoubleUscores . ensureLeadAlpha . map steam
 where ensureLeadAlpha v@(c:cs)
        | isAlpha c  = v
        | otherwise  = "identifier_"++v
       killDoubleUscores [] = []
       killDoubleUscores ('_':'_':r) = killDoubleUscores $ '_':r
       killDoubleUscores (c:cs) = c : killDoubleUscores cs
       steam c | isAlphaNum c  = c
               | otherwise     = '_'



exportCXXinliningHeader :: FilePath -> CXXCode() -> IO()
exportCXXinliningHeader hFile code = do
   hash <- randomIO
   writeFile hFile . cxxCodeString $ envIncludeGuards hash
 where envIncludeGuards :: Word64 -> CXXCode()
       envIncludeGuards hash = do
          cxxLine $ "#ifndef "++inclGuardKey
          cxxLine $ "#define "++inclGuardKey
          code
          cxxLine $ "#endif"
        where inclGuardKey = guardOk_hname++"_"++showHex hash[]
              guardOk_hname = map toUpper $ makeSafeCXXIdentifier hFile
