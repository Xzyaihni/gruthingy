use std::{
    fs,
    path::Path
};


pub fn process_includes(parent_path: &Path, text: String) -> String
{
    let include_prefix = "#include";

    if !text.contains(include_prefix)
    {
        return text;
    }

    text.lines().map(|line|
    {
        if let Some(include_path) = line.trim_start().strip_prefix(include_prefix)
        {
            let include_path = include_path.trim();
            let include_path = &include_path[1..(include_path.len() - 1)];
            let full_include_path = parent_path.join(include_path);

            process_includes(parent_path, fs::read_to_string(&full_include_path).unwrap_or_else(|err|
            {
                panic!("error trying to include file {}: {err}", full_include_path.display())
            }))
        } else
        {
            line.to_owned()
        }
    }).reduce(|acc, x| acc + "\n" + x.as_ref()).unwrap_or_default()
}
