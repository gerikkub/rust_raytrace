
use std::io;
use std::time;
use std::usize;
use std::collections::HashMap;
use crossterm::{execute, terminal, style, cursor};

#[derive(Clone, Copy)]
pub enum ProgressStat {
    Time(time::Duration),
    Count(usize)
}

impl ProgressStat {
    pub fn as_time(&self) -> &time::Duration {
        match self {
            ProgressStat::Time(t) => &t,
            _ => panic!("Value Not Time")
        }
    }

    pub fn as_time_mut<'a>(&'a mut self) -> &'a mut time::Duration {
        match self {
            ProgressStat::Time(t) => t,
            _ => panic!("Value Not Time")
        }
    }

    pub fn as_count(&self) -> &usize {
        match self {
            ProgressStat::Count(t) => &t,
            _ => panic!("Value Not Count")
        }
    }

    pub fn as_count_mut(&mut self) -> &mut usize {
        match self {
            ProgressStat::Count(t) => t,
            _ => panic!("Value Not Count")
        }
    }
}

pub struct ProgressCtx {
    start_time: time::Instant,
    stop_time: time::Instant,
    runtimes: HashMap<String, ProgressStat>,
    num_threads: usize,
    width: usize,
    height: usize,
    finished_pixels: usize,
    total_rays: usize,
    enable_io: bool
}

pub fn create_ctx(threads: usize, width: usize, height: usize, enable_io: bool) -> ProgressCtx {
    if enable_io {
        execute!(
            io::stdout(),
            terminal::EnterAlternateScreen,
            terminal::Clear(terminal::ClearType::All),
            cursor::Hide
        ).unwrap();
    }

    ProgressCtx {
        start_time: time::Instant::now(),
        stop_time: time::Instant::now(),
        runtimes: HashMap::new(),
        num_threads: threads,
        width: width,
        height: height,
        finished_pixels: 0,
        total_rays: 0,
        enable_io: enable_io
    }
}


impl ProgressCtx {

    pub fn update(&mut self, tnum: usize, row: usize, pixels: usize, runstats: &HashMap<String, ProgressStat>) {

        let elapsed = time::Instant::now() - self.start_time;
        let secs = elapsed.as_secs();
        let sub_millis = elapsed.subsec_millis();

        self.finished_pixels += pixels;
        if runstats.contains_key("Rays") {
            self.total_rays += runstats.get("Rays").unwrap().as_count();
        }
        let total_pixels = self.width * self.height;

        let complete_fraction = (self.finished_pixels as f64)/(total_pixels as f64);
        let secs_per_pixel = (secs as f64) / (self.finished_pixels as f64);
        let remaining_sec_est = (secs_per_pixel * ((total_pixels - self.finished_pixels) as f64)) as usize;

        let rays_per_sec = self.total_rays as f64 / elapsed.as_secs_f64();

        if self.enable_io {
            execute!(io::stdout(),
                        cursor::MoveTo(0, 0),
                        style::Print(format!("Run time: {}:{:02}.{:03} Est: {}:{:02}\n",
                                            secs/60, secs % 60, sub_millis,
                                            remaining_sec_est/60, remaining_sec_est % 60)),
                        style::Print(format!("Completed: {}/{} {:.2}\n",
                                             self.finished_pixels, total_pixels,
                                             complete_fraction * 100.)),
                        style::Print(format!("Rays so far: {:.3} million {:.3} million rays/s\n",
                                             self.total_rays as f64/1_000_000., rays_per_sec/1_000_000.)),
                        style::Print(format!("Threads: {}", self.num_threads)),

                        cursor::MoveTo(0, (tnum+4) as u16),
                        terminal::Clear(terminal::ClearType::CurrentLine),
                        style::Print(format!("Thread {:02}: {}", tnum, row))
                    ).unwrap();
        }

        for (k, v) in runstats {
            match &v {
                ProgressStat::Time(t) => {
                    *self.runtimes.entry(k.to_string()).or_insert(ProgressStat::Time(time::Duration::from_nanos(0))).as_time_mut() += *t;
                },
                ProgressStat::Count(c) => {
                    *self.runtimes.entry(k.to_string()).or_insert(ProgressStat::Count(0)).as_count_mut() += *c;
                }
            }
        }
    }

    pub fn finish(&mut self) {

        self.stop_time = time::Instant::now();

        if self.enable_io {
            execute!(
                io::stdout(),
                terminal::LeaveAlternateScreen,
                cursor::Show
            ).unwrap();
        }
    }

    pub fn print_stats(&self) {
        println!("Processed {:.3} million rays in {:.3} seconds. {:.3} million rays/s",
                 self.total_rays as f64 / 1_000_000.,
                 (self.stop_time - self.start_time).as_millis() as f64 / 1000.,
                 (self.total_rays as f64 * 1000.) / (self.stop_time - self.start_time).as_millis() as f64 / 1_000_000.
                );
        for (k, v) in &self.runtimes {
            match v {
                ProgressStat::Time(d) => println!("{}: {}.{:0>6}", k, d.as_secs(), d.subsec_micros()),
                ProgressStat::Count(c) => println!("{}: {}", k, c)
            }
        }

    }

}