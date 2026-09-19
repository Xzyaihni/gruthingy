import Data.Maybe;


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

parseData :: String -> [Float]
parseData dataText = read dataText

parseOperand :: String -> DebugOperand
parseOperand operand = let (name, afterName) = break (\c -> c == ':') operand
                           (variableNameRaw, dataRaw) = break (\c -> c == '=') afterName
                       in DebugOperand
                            name
                            (drop 2 (take ((length variableNameRaw) - 1) variableNameRaw))
                            (parseData (drop 2 dataRaw))

ignoreInsideBrackets :: Char -> Char -> (Bool -> Char -> Bool)
ignoreInsideBrackets start end = (\state c -> if (c == start) then False else (if (c == end) then True else state))

parseOperands :: String -> [DebugOperand]
parseOperands [] = []
parseOperands operands = let operandsSplit = splitByUnless (ignoreInsideBrackets '[' ']') id True ',' operands
                         in map parseOperand $ map (dropWhile (\c -> c == ' ')) operandsSplit

beforeOperandsState :: (Bool, Bool) -> Char -> (Bool, Bool)
beforeOperandsState (aState, bState) c = (if (c == '=') then True else (if (c == ')') then False else aState), ((ignoreInsideBrackets '{' '}') bState c))

beforeOperandsDecide :: (Bool, Bool) -> Bool
beforeOperandsDecide (a, b) = a && b

splitUpOperationRaw :: String -> (String, String, String)
splitUpOperationRaw operation = let nameRaw = takeWhile (\x -> x /= '(') operation
                                    operationBefore = drop ((length nameRaw) + 1 + (length "BEFORE ")) operation
                                    (beforeOperands:afterOperandsStart) = splitByUnless beforeOperandsState beforeOperandsDecide (False, True) ')' operationBefore
                                    afterOperands = drop (length " AFTER ") (head afterOperandsStart)
                                in ((take ((length nameRaw) - 1) nameRaw), beforeOperands, afterOperands)

parseOperationWithIndex :: Int -> String -> DebugOperation
parseOperationWithIndex lineIndex operation = let (name, beforeOperands, afterOperands) = splitUpOperationRaw operation
                                    in DebugOperation lineIndex name (parseOperands beforeOperands) (parseOperands afterOperands)

parseOperation :: String -> DebugOperation
parseOperation = parseOperationWithIndex 0

validOperation :: String -> Bool
validOperation = (== ' ') . last . (takeWhile (\c -> c /= '('))

parseOperations :: String -> [DebugOperation]
parseOperations operations = map (\(x, i) -> parseOperationWithIndex i x) $ filter (\(x, _) -> validOperation x) $ (zip (lines operations) [0..])

operandValues :: DebugOperand -> [Float]
operandValues (DebugOperand name variable values) = values

operationInputs :: DebugOperation -> [DebugOperand]
operationInputs (DebugOperation lineIndex name inputs outputs) = inputs

operationOutputs :: DebugOperation -> [DebugOperand]
operationOutputs (DebugOperation lineIndex name inputs outputs) = outputs

operationLineIndex :: DebugOperation -> Int
operationLineIndex (DebugOperation lineIndex name inputs outputs) = lineIndex

isOutputMatch :: [Float] -> [Float] -> Bool
isOutputMatch a b = a == b

isOutputsMatch :: ([Float] -> [Float] -> Bool) -> [[Float]] -> [[Float]] -> Bool
isOutputsMatch matcher a b = if (length a) /= (length b)
                        then False
                        else all id $ map (\(ai, bi) -> matcher ai bi) $ zip a b

operandsWithLine :: String -> [([[Float]], DebugOperation)]
operandsWithLine s = map (\x -> ((map operandValues) $ operationOutputs x, x)) $ parseOperations s

findMismatchOutputWith :: ([Float] -> [Float] -> Bool) -> String -> String -> Maybe (DebugOperation, DebugOperation)
findMismatchOutputWith matcher aInput bInput = let f = operandsWithLine
                                                   p = lines
                                               in fmap (\((_, a), (_, b)) -> (a, b)) $ listToMaybe $ filter (\((a, _), (b, _)) -> not (isOutputsMatch matcher a b)) $ zip (f aInput) (f bInput)

findMismatchOutput :: String -> String -> Maybe (DebugOperation, DebugOperation)
findMismatchOutput a b = findMismatchOutputWith isOutputMatch a b
