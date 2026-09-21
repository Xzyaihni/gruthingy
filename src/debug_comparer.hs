import Data.Maybe;
import Data.List;
import Control.Monad;
import Debug.Trace;
import Text.Read;
import Data.Either;


data DebugOperand = DebugOperand String String [Float] deriving Show;

data DebugOperation = DebugOperation Int String [DebugOperand] [DebugOperand] deriving Show;

splitBy :: Eq a => a -> [a] -> [[a]]
splitBy separator text = let (before, after) = break (\x -> x == separator) text
                             in if (null after) then [before] else (before : (splitBy separator (drop 1 after)))

splitByUnlessStateful :: Eq a => (s -> a -> s) -> (s -> Bool) -> s -> a -> [a] -> [a] -> [[a]]
splitByUnlessStateful splitStateGetter decide state separator [] current = [(reverse current)]
splitByUnlessStateful splitStateGetter decide state separator (c:text) current = let newState = splitStateGetter state c
                                                                                 in if ((decide state) && (c == separator))
                                                                                    then ((reverse current) : (splitByUnlessStateful splitStateGetter decide newState separator text []))
                                                                                    else splitByUnlessStateful splitStateGetter decide newState separator text (c : current)

splitByUnless :: Eq a => (s -> a -> s) -> (s -> Bool) -> s -> a -> [a] -> [[a]]
splitByUnless splitStateGetter decide startState separator text = splitByUnlessStateful splitStateGetter decide startState separator text []

parseData :: String -> Either String [Float]
parseData [] = Left "empty string"
parseData dataText = fromMaybe (Left ("error parsing " ++ dataText))
                      $ sequence
                      $ Right
                      $ (if (head dataText) /= '[' then (fmap (\x -> [x]) $ readMaybe dataText) else (readMaybe dataText))

parseOperand :: String -> Either String DebugOperand
parseOperand operand = let (name, afterName) = break (\c -> c == ':') operand
                           (variableNameRaw, dataRaw) = break (\c -> c == '=') afterName
                           values = (parseData (drop 2 dataRaw))
                       in values >>= \values -> Right $ DebugOperand
                            name
                            (drop 2 (take ((length variableNameRaw) - 1) variableNameRaw))
                            values

ignoreInsideBrackets :: Char -> Char -> (Bool -> Char -> Bool)
ignoreInsideBrackets start end = (\state c -> if (c == start) then False else (if (c == end) then True else state))

parseOperands :: String -> Either String [DebugOperand]
parseOperands [] = Right []
parseOperands operands = let operandsSplit = splitByUnless (ignoreInsideBrackets '[' ']') id True ',' operands
                         in sequence $ map parseOperand $ map (dropWhile (\c -> c == ' ')) operandsSplit

beforeOperandsState :: (Bool, Bool) -> Char -> (Bool, Bool)
beforeOperandsState (aState, bState) c = (if (c == '=') then True else (if (c == ')') || (c == '(') then False else aState), ((ignoreInsideBrackets '{' '}') bState c))

beforeOperandsDecide :: (Bool, Bool) -> Bool
beforeOperandsDecide (a, b) = a && b

splitUpOperationRaw :: String -> (String, String, String)
splitUpOperationRaw operation = let nameRaw = takeWhile (\x -> x /= '(') operation
                                    operationBefore = drop ((length nameRaw) + 1 + (length "BEFORE ")) operation
                                    (beforeOperands:afterOperandsStart) = splitByUnless beforeOperandsState beforeOperandsDecide (False, True) ')' operationBefore
                                    afterOperands = drop (length " AFTER ") (head afterOperandsStart)
                                in ((take ((length nameRaw) - 1) nameRaw), beforeOperands, afterOperands)

unwrapErrorAt :: Int -> Either String a -> a
unwrapErrorAt index (Left err) = error ("on line " ++ (show index) ++ ": " ++ err)
unwrapErrorAt index (Right a) = a

parseOperationWithIndex :: Int -> String -> DebugOperation
parseOperationWithIndex lineIndex operation = let (name, beforeOperands, afterOperands) = splitUpOperationRaw operation
                                                  before = unwrapErrorAt lineIndex (parseOperands beforeOperands)
                                                  after = unwrapErrorAt lineIndex (parseOperands afterOperands)
                                              in DebugOperation lineIndex name before after

parseOperation :: String -> DebugOperation
parseOperation = parseOperationWithIndex 0

validOperation :: String -> Bool
validOperation = (== ' ') . last . (takeWhile (\c -> c /= '('))

parseOperations :: String -> [DebugOperation]
parseOperations operations = map (\(x, i) -> parseOperationWithIndex i x) $ filter (\(x, _) -> validOperation x) $ (zip (lines operations) [0..])

operandValues :: DebugOperand -> [Float]
operandValues (DebugOperand name variable values) = values

type OperandsSelector = DebugOperation -> [DebugOperand]

operationInputs :: OperandsSelector
operationInputs (DebugOperation lineIndex name inputs outputs) = inputs

operationOutputs :: OperandsSelector
operationOutputs (DebugOperation lineIndex name inputs outputs) = outputs

operationLineIndex :: DebugOperation -> Int
operationLineIndex (DebugOperation lineIndex name inputs outputs) = lineIndex

isOutputMatch :: [Float] -> [Float] -> Bool
isOutputMatch a b = a == b

type OutputsValues = [[Float]]

isOutputsMatch :: ([Float] -> [Float] -> Bool) -> OutputsValues -> OutputsValues -> Bool
isOutputsMatch matcher a b = if (length a) /= (length b)
                        then False
                        else all id $ map (\(ai, bi) -> matcher ai bi) $ zip a b

type OutputsValuesWithOp = (OutputsValues, DebugOperation)

operandsWithOp :: OperandsSelector -> [String] -> [[OutputsValuesWithOp]]
operandsWithOp inputs s = map (\operations -> map (\x -> ((map operandValues) $ inputs x, x)) operations) $ (map parseOperations s)

zipOperationInfos :: OperandsSelector -> [String] -> String -> [([OutputsValuesWithOp], OutputsValuesWithOp)]
zipOperationInfos inputs aInput bInput = zip (transpose $ operandsWithOp inputs aInput) (head $ operandsWithOp inputs [bInput])

type MatcherType = ([OutputsValuesWithOp], OutputsValuesWithOp) -> Bool

oneToOneMatcher :: MatcherType
oneToOneMatcher = undefined

keepUnmatching :: MatcherType -> [([OutputsValuesWithOp], OutputsValuesWithOp)] -> [([OutputsValuesWithOp], OutputsValuesWithOp)]
keepUnmatching matcher = filter (not . matcher)

findMismatchWith :: OperandsSelector -> MatcherType -> [String] -> String -> [([DebugOperation], DebugOperation)]
findMismatchWith inputs matcher aInput bInput = map (\(a, (_, b)) -> (map snd a, b))
                                                 $ keepUnmatching matcher
                                                 $ zipOperationInfos inputs aInput bInput

findMismatchOutput :: String -> String -> Maybe (DebugOperation, DebugOperation)
findMismatchOutput a b = listToMaybe $ fmap (\(a, b) -> (head a, b)) $ findMismatchWith operationOutputs oneToOneMatcher [a] b

batchMatchSingleOutput :: ([Float], [Float]) -> [Float] -> Bool
batchMatchSingleOutput (firstA, secondA) b = if (length firstA) == (length b)
                                                then (if (firstA == secondA) then firstA == b else (map (\(a, b) -> a + b) $ zip firstA secondA) == b)
                                                else (firstA == (take (length firstA) b)) && (secondA == (drop (length firstA) b))

batchMatcher :: MatcherType
batchMatcher (aPair, (b, bOp)) = if (length b) /= (length $ fst $ head aPair)
                                  then error ("outputs amount doesnt match (in " ++ (show bOp) ++ " vs " ++ (show aPair) ++ "): " ++ (show $ length $ fst $ head aPair) ++ " vs " ++ (show $ length b))
                                  else let firstA = (aPair !! 0)
                                           secondA = (aPair !! 1)
                                           f = fst
                                       in all id $ map (\(a, b) -> batchMatchSingleOutput a b) $ zip (zip (f firstA) (f secondA)) b

writeUnmatchingBatchesContents :: OperandsSelector -> String -> String -> String
writeUnmatchingBatchesContents inputs unbatched batched = let batchedLines = length $ lines batched
                                                   in unlines
                                                       $ map (\(a, b) -> unlines $ (map (\x -> "  " ++ x) $ [show (a !! 0), show (a !! 1)]) ++ [show b])
                                                       $ findMismatchWith
                                                          inputs
                                                          batchMatcher
                                                          [(unlines $ take batchedLines $ lines unbatched), (unlines $ drop batchedLines $ lines unbatched)]
                                                          batched

writeUnmatchingBatches :: OperandsSelector -> String -> String -> String -> IO ()
writeUnmatchingBatches inputs unbatchedPath batchedPath outputPath = (readFile batchedPath) >>=
    \batched -> join $ fmap (\unbatched -> (writeFile outputPath (writeUnmatchingBatchesContents inputs unbatched batched))) (readFile unbatchedPath)
