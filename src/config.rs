use std::{
    env,
    process,
    fs::File,
    fmt::{self, Display},
    collections::HashSet,
    num::{ParseIntError, ParseFloatError}
};

use crate::complain;


enum ArgError
{
    Parse(String),
    EnumParse{value: String, all: String},
    UnexpectedArg(String),
    DuplicateArg(String),
    MissingValue(String)
}

impl Display for ArgError
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{}", match self
        {
            Self::Parse(x) => format!("error parsing {x}"),
            Self::EnumParse{value: x, all} => format!("error parsing {x}, available options: {all}"),
            Self::UnexpectedArg(x) => format!("unexpected argument {x}"),
            Self::DuplicateArg(x) => format!("duplicate argument {x}"),
            Self::MissingValue(x) => format!("missing value after {x} argument")
        })
    }
}

impl From<(&str, ParseIntError)> for ArgError
{
    fn from(value: (&str, ParseIntError)) -> Self
    {
        Self::Parse(value.0.to_owned())
    }
}

impl From<(&str, ParseFloatError)> for ArgError
{
    fn from(value: (&str, ParseFloatError)) -> Self
    {
        Self::Parse(value.0.to_owned())
    }
}

enum ArgType
{
    Variable,
    Flag(bool),
    Help
}

#[derive(Debug)]
enum ArgParseInfo
{
    Variable(String),
    Flag(bool)
}

impl ArgParseInfo
{
    pub fn flag(self) -> bool
    {
        match self
        {
            Self::Flag(x) => x,
            x => panic!("incorrect value: {x:?}")
        }
    }

    pub fn variable(self) -> String
    {
        match self
        {
            Self::Variable(x) => x,
            x => panic!("incorrect value: {x:?}")
        }
    }
}

struct ArgInfo<'a>
{
    value: Option<&'a mut dyn ArgParsable>,
    short: Option<char>,
    long: String,
    description: String,
    kind: ArgType,
    encountered: bool
}

impl<'a> ArgInfo<'a>
{
    pub fn help(&self, longest_arg: usize) -> String
    {
        let head = self.help_head();

        // this technically would overpad if the longest arg isnt a variable but wutever
        // i dont rly care
        let padded = longest_arg + "-a, --=VALUE".len();

        format!(" {head:<padded$}  {}", self.description)
    }

    fn help_head(&self) -> String
    {
        let mut line = match self.short
        {
            Some(c) => format!("-{c},"),
            None => "   ".to_owned()
        };

        line += &format!(" --{}", self.long);

        match self.kind
        {
            ArgType::Variable =>
            {
                line += "=VALUE";
            },
            _ => ()
        }

        line
    }
}

struct ArgParser<'a>
{
    args: Vec<ArgInfo<'a>>
}

impl<'a> ArgParser<'a>
{
    pub fn new() -> Self
    {
        Self{args: Vec::new()}
    }

    pub fn push(
        &mut self,
        value: &'a mut dyn ArgParsable,
        short: impl Into<Option<char>>,
        long: impl Into<String>,
        description: impl Into<String>
    )
    {
        self.args.push(ArgInfo{
            value: Some(value),
            short: short.into(),
            long: long.into(),
            description: description.into(),
            kind: ArgType::Variable,
            encountered: false
        });
    }

    pub fn push_flag(
        &mut self,
        value: &'a mut dyn ArgParsable,
        short: impl Into<Option<char>>,
        long: impl Into<String>,
        description: impl Into<String>,
        state: bool
    )
    {
        self.args.push(ArgInfo{
            value: Some(value),
            short: short.into(),
            long: long.into(),
            description: description.into(),
            kind: ArgType::Flag(state),
            encountered: false
        });
    }

    pub fn parse(mut self, mut args: impl Iterator<Item=String>) -> Result<(), ArgError>
    {
        self.args.push(ArgInfo{
            value: None,
            short: Some('h'),
            long: "help".to_owned(),
            description: "shows this message".to_owned(),
            kind: ArgType::Help,
            encountered: false
        });

        self.validate();

        while let Some(raw_arg) = args.next()
        {
            if raw_arg.starts_with("--")
            {
                let arg = &raw_arg[2..];

                if let Some(found) = self.args.iter_mut().find(|this_arg| this_arg.long == arg)
                {
                    match found.kind
                    {
                        ArgType::Help => self.print_help(),
                        _ => ()
                    }

                    Self::on_arg(&mut args, found, &raw_arg)?;
                } else
                {
                    return Err(ArgError::UnexpectedArg(raw_arg));
                }
            } else if raw_arg.starts_with('-')
            {
                let arg = &raw_arg[1..];

                if arg.len() != 1
                {
                    return Err(ArgError::UnexpectedArg(raw_arg));
                }

                let c = arg.chars().next().unwrap();

                if let Some(found) = self.args.iter_mut().find(|arg| arg.short == Some(c))
                {
                    match found.kind
                    {
                        ArgType::Help => self.print_help(),
                        _ => ()
                    }

                    Self::on_arg(&mut args, found, &raw_arg)?;
                } else
                {
                    return Err(ArgError::UnexpectedArg(raw_arg));
                }
            } else
            {
                return Err(ArgError::UnexpectedArg(raw_arg));
            }
        }

        Ok(())
    }

    fn print_help(self) -> !
    {
        println!("usage: {} [args]", env::args().next().unwrap());

        let longest_arg = self.args.iter().map(|arg| arg.long.len()).max()
            .unwrap_or(0);

        for arg in self.args
        {
            println!("{}", arg.help(longest_arg));
        }

        process::exit(0)
    }

    fn on_arg(
        mut args: impl Iterator<Item=String>,
        arg: &mut ArgInfo,
        arg_value: &str
    ) -> Result<(), ArgError>
    {
        if arg.encountered
        {
            return Err(ArgError::DuplicateArg(arg_value.to_owned()));
        }

        arg.encountered = true;

        let info = match arg.kind
        {
            ArgType::Variable =>
            {
                let value = args.next().ok_or_else(||
                {
                    ArgError::MissingValue(arg_value.to_owned())
                })?;

                ArgParseInfo::Variable(value)
            },
            ArgType::Flag(x) => ArgParseInfo::Flag(x),
            ArgType::Help => unreachable!()
        };

        arg.value.as_mut().unwrap().parse(info)?;

        Ok(())
    }

    fn validate(&self)
    {
        let short_args = self.args.iter()
            .filter_map(|arg| arg.short.as_ref());

        let c_set: HashSet<&char> = short_args.clone().collect();
        assert_eq!(c_set.len(), short_args.count());
    }
}

trait ParsableInner
where
    Self: Sized
{
    fn parse_inner(value: &str) -> Result<Self, ArgError>;
}

trait ParsableEnum
{
    type Iter: Iterator<Item=Self>;


    fn iter() -> Self::Iter;
    fn as_str(&self) -> &'static str;
}

macro_rules! iterable_enum
{
    ($enum_name:ident, $($key:ident, $name:ident),+) =>
    {
        pub enum $enum_name
        {
            $($key,)+
        }

        impl $enum_name
        {
            const fn len() -> usize
            {
                [
                    $(stringify!($key),)+
                ].len()
            }
        }

        impl ParsableEnum for $enum_name
        {
            type Iter = std::array::IntoIter<Self, { Self::len() }>;


            fn iter() -> Self::Iter
            {
                [
                    $(Self::$key,)+
                ].into_iter()
            }

            fn as_str(&self) -> &'static str
            {
                match self
                {
                    $(Self::$key => stringify!($name),)+
                }
            }
        }
    }
}

iterable_enum!
{
    ProgramMode,
    Train, train,
    Run, run,
    Test, test,
    CreateDictionary, create_dictionary,
    TrainEmbeddings, train_embeddings,
    WeightsImage, weights_image
}

impl<T: ParsableEnum> ParsableInner for T
{
    fn parse_inner(value: &str) -> Result<Self, ArgError>
    {
        let value = value.to_lowercase();

        Self::iter().find(|x| x.as_str() == value)
            .ok_or_else(||
            {
                let all = Self::iter().map(|x| x.as_str().to_owned()).reduce(|acc, x|
                {
                    acc + ", " + &x
                }).unwrap_or_else(String::new);

                ArgError::EnumParse{value: value.to_owned(), all}
            })
    }
}

impl ParsableInner for String
{
    fn parse_inner(value: &str) -> Result<Self, ArgError>
    {
        Ok(value.to_owned())
    }
}

impl ParsableInner for usize
{
    fn parse_inner(value: &str) -> Result<Self, ArgError>
    {
        value.parse::<usize>().map_err(|err| (value, err).into())
    }
}

impl ParsableInner for f32
{
    fn parse_inner(value: &str) -> Result<Self, ArgError>
    {
        value.parse::<f32>().map_err(|err| (value, err).into())
    }
}

trait ArgParsable
{
    fn parse(&mut self, value: ArgParseInfo) -> Result<(), ArgError>;
}

impl ArgParsable for bool
{
    fn parse(&mut self, value: ArgParseInfo) -> Result<(), ArgError>
    {
        *self = value.flag();

        Ok(())
    }
}

impl<T: ParsableInner> ArgParsable for T
{
    fn parse(&mut self, value: ArgParseInfo) -> Result<(), ArgError>
    {
        *self = T::parse_inner(&value.variable())?;

        Ok(())
    }
}

impl<T: ParsableInner> ArgParsable for Option<T>
{
    fn parse(&mut self, value: ArgParseInfo) -> Result<(), ArgError>
    {
        *self = Some(T::parse_inner(&value.variable())?);

        Ok(())
    }
}

pub struct Config
{
    pub iterations: usize,
    pub batch_size: usize,
    pub hidden_size: usize,
    pub layers_amount: usize,
    pub steps_num: usize,
    pub embeddings_size: usize,
    pub learning_rate: Option<f32>,
    pub calculate_loss: bool,
    pub calculate_accuracy: bool,
    pub testing_data: Option<String>,
    pub network_path: String,
    pub input: Option<String>,
    pub output: Option<String>,
    pub tokens_amount: usize,
    pub temperature: f32,
    pub replace_invalid: bool,
    pub less_info: bool,
    pub mode: ProgramMode,
    pub dictionary_path: String
}

impl Config
{
    pub fn parse(args: impl Iterator<Item=String>) -> Self
    {
        let mut iterations = 1;
        let mut batch_size = 32;
        let mut hidden_size = 512;
        let mut layers_amount = 3;
        let mut steps_num = 64;
        let mut embeddings_size = 64;
        let mut learning_rate = None;
        let mut calculate_loss = true;
        let mut calculate_accuracy = false;
        let mut testing_data = None;
        let mut network_path = "network.nn".to_owned();
        let mut input = None;
        let mut output = None;
        let mut tokens_amount = 100;
        let mut temperature = 1.0;
        let mut replace_invalid = true;
        let mut dictionary_path = "dictionary.txt".to_owned();
        let mut less_info = false;
        let mut mode = None;

        let mut parser = ArgParser::new();

        parser.push(&mut iterations, 'I', "iterations", "the amount of iterations to train for");
        parser.push(&mut batch_size, 'b', "batch", "minibatch size");
        parser.push(&mut hidden_size, None, "hidden", "hidden layers size");
        parser.push(&mut layers_amount, None, "layers", "amount of hidden layers");
        parser.push(&mut steps_num, 's', "steps", "amount of timesteps the network remembers");
        parser.push(&mut embeddings_size, 'e', "embeddings", "size of the embeddings vector");
        parser.push(&mut learning_rate, 'l', "learning-rate", "learning rate for the optimizer");
        parser.push_flag(&mut calculate_accuracy, 'a', "accuracy", "calculate accuracy", true);
        parser.push_flag(&mut calculate_loss, None, "no-loss", "dont calculate loss", false);
        parser.push(&mut testing_data, 't', "testing", "data for calculating the loss/accuracy");
        parser.push(&mut network_path, 'p', "path", "path to the network");
        parser.push(&mut input, 'i', "input", "input");
        parser.push(&mut output, 'o', "output", "output path");
        parser.push(&mut tokens_amount, 'n', "number", "number of tokens to generate");
        parser.push(&mut temperature, 'T', "temperature", "softmax temperature");
        parser.push_flag(&mut replace_invalid, 'r', "raw", "dont replace invalid utf8", false);
        parser.push_flag(&mut less_info, None, "less-info", "display less info when training", true);
        parser.push(&mut mode, 'm', "mode", "program mode");
        parser.push(&mut dictionary_path, 'd', "dictionary", "path to the dictionary");

        parser.parse(args).unwrap_or_else(|err|
        {
            complain(err.to_string())
        });

        let mode = mode.unwrap_or_else(||
        {
            let modes = ProgramMode::iter()
                .map(|x| x.as_str().to_owned())
                .reduce(|acc, x|
                {
                    acc + ", " + &x
                })
                .unwrap_or_else(String::new);

            complain(format!("provide a valid mode: {modes}"))
        });

        Self{
            iterations,
            batch_size,
            hidden_size,
            layers_amount,
            steps_num,
            embeddings_size,
            learning_rate,
            calculate_loss,
            calculate_accuracy,
            testing_data,
            network_path,
            input,
            output,
            tokens_amount,
            temperature,
            replace_invalid,
            dictionary_path,
            less_info,
            mode
        }
    }

    pub fn get_input(&self) -> &str
    {
        self.input.as_ref().unwrap_or_else(||
        {
            complain("plz provide the input (-i or --input)")
        })
    }

    pub fn get_input_file(&self) -> File
    {
        let input = self.get_input();
        File::open(input)
            .unwrap_or_else(|err|
            {
                complain(format!("give a valid file plz, cant open {input} ({err})"))
            })
    }
}
