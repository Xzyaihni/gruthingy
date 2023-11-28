use std::{
    env,
    process,
    fmt::{self, Display},
    collections::HashSet,
    num::{ParseIntError, ParseFloatError}
};

use crate::complain;


enum ArgError
{
    Parse(String),
    UnexpectedArg(String),
    MissingValue(String)
}

impl Display for ArgError
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{}", match self
        {
            Self::Parse(x) => format!("error parsing {x}"),
            Self::UnexpectedArg(x) => format!("unexpected argument {x}"),
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
    kind: ArgType
}

impl<'a> ArgInfo<'a>
{
    pub fn help(&self, longest_arg: usize) -> String
    {
        let head = self.help_head();

        let padded = longest_arg + "-a,".len();

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
            kind: ArgType::Variable
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
            kind: ArgType::Flag(state)
        });
    }

    pub fn parse(mut self, mut args: impl Iterator<Item=String>) -> Result<(), ArgError>
    {
        self.args.push(ArgInfo{
            value: None,
            short: Some('h'),
            long: "help".to_owned(),
            description: "shows this message".to_owned(),
            kind: ArgType::Help
        });

        self.validate();

        while let Some(raw_arg) = args.next()
        {
            if raw_arg.starts_with('-')
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
            } else if raw_arg.starts_with("--")
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
    pub steps_num: usize,
    pub learning_rate: Option<f32>,
    pub calculate_loss: bool,
    pub calculate_accuracy: bool,
    pub testing_data: Option<String>,
    pub network_path: String,
    pub dictionary_path: String
}

impl Config
{
    pub fn parse(args: impl Iterator<Item=String>) -> Self
    {
        let mut iterations = 1;
        let mut batch_size = 32;
        let mut hidden_size = 512;
        let mut steps_num = 64;
        let mut learning_rate = None;
        let mut calculate_loss = true;
        let mut calculate_accuracy = false;
        let mut testing_data = None;
        let mut network_path = "network.nn".to_owned();
        let mut dictionary_path = "dictionary.txt".to_owned();

        let mut parser = ArgParser::new();

        parser.push(&mut iterations, 'i', "iterations", "the amount of iterations to train for");
        parser.push(&mut batch_size, 'b', "batch", "minibatch size");
        parser.push(&mut hidden_size, None, "hidden", "hidden layers size");
        parser.push(&mut steps_num, 's', "steps", "amount of timesteps the network remembers");
        parser.push(&mut learning_rate, 'l', "learning-rate", "learning rate for the optimizer");
        parser.push_flag(&mut calculate_accuracy, 'a', "accuracy", "calculate accuracy", true);
        parser.push_flag(&mut calculate_loss, None, "no-loss", "dont calculate loss", false);
        parser.push(&mut testing_data, 't', "testing", "data for calculating the loss/accuracy");
        parser.push(&mut network_path, 'p', "path", "path to the network");
        parser.push(&mut dictionary_path, 'd', "dictionary", "path to the dictionary");

        parser.parse(args).unwrap_or_else(|err|
        {
            complain(err.to_string())
        });

        Self{
            iterations,
            batch_size,
            hidden_size,
            steps_num,
            learning_rate,
            calculate_loss,
            calculate_accuracy,
            testing_data,
            network_path,
            dictionary_path
        }
    }
}
