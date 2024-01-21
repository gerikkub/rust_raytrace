
use std::io;
use std::time;
use std::time::Instant;
use std::usize;
use crossterm::{execute, terminal, style, cursor};

pub struct ProgressCtx {
    start_time: time::Instant,
    stop_time: time::Instant,
    etas: Vec<usize>,
    total_rays: usize,
    enable_io: bool
}

pub fn create_ctx(threads: usize, total_rays: usize, enable_io: bool) -> ProgressCtx {
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
        etas: vec![0; threads],
        total_rays: total_rays,
        enable_io: enable_io
    }
}


impl ProgressCtx {

    pub fn update(&mut self, tnum: usize, rays: usize, total_rays: usize) {

        let elapsed = time::Instant::now() - self.start_time;
        let secs = elapsed.as_secs();
        let sub_millis = elapsed.subsec_millis();

        let complete_fraction = (rays as f64)/(total_rays as f64);
        let secs_per_ray = (secs as f64) / (rays as f64);
        let remaining_sec_est = (secs_per_ray * ((total_rays - rays) as f64)) as usize;
        self.etas[tnum] = remaining_sec_est;

        let max_eta = self.etas.iter().max().unwrap();

        if self.enable_io {
            execute!(io::stdout(),
                        cursor::MoveTo(0, 0),
                        style::Print(format!("Run time: {}:{:02}.{:03} ETA: {}:{:02}",
                                            secs/60, secs % 60, sub_millis,
                                            max_eta / 60, max_eta % 60))
                    ).unwrap();


            execute!(io::stdout(),
                        cursor::MoveTo(0, (tnum+1) as u16),
                        style::Print(format!("Thread {:02}: {}/{} {:.2} ETA: {}:{:02}",
                                            tnum, rays, total_rays,
                                            100. * complete_fraction,
                                            remaining_sec_est / 60,
                                            remaining_sec_est % 60,
                                            ))
                    ).unwrap();
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
        println!("Processed {} rays in {:.3} seconds. {:.0} rays/s",
                 self.total_rays,
                 (self.stop_time - self.start_time).as_millis() as f64 / 1000.,
                 (self.total_rays as f64 * 1000.) / (self.stop_time - self.start_time).as_millis() as f64
                );

    }

}
