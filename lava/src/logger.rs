use log::Level;
use pretty_env_logger::env_logger::fmt::{Color, Style, StyledValue};

pub fn init_logger() {
    pretty_env_logger::formatted_builder()
        .parse_default_env()
        .format(|f, record| {
            use std::io::Write;
            let mut style = f.style();
            let level = colored_level(&mut style, record.level());
            writeln!(f, " {} > {}", level, record.args(),)
        })
        .init();
}

fn colored_level(style: &mut Style, level: Level) -> StyledValue<'_, &'static str> {
    match level {
        Level::Trace => style.set_color(Color::Magenta).value("TRACE"),
        Level::Debug => style.set_color(Color::Blue).value("DEBUG"),
        Level::Info => style.set_color(Color::Green).value("INFO "),
        Level::Warn => style.set_color(Color::Yellow).value("WARN "),
        Level::Error => style.set_color(Color::Red).value("ERROR"),
    }
}
