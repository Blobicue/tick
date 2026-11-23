//! A minimal persistent fork-join pool: lock-free job slots
//! (AtomicPtr) with a syscall-free fast path, short spin then condvar
//! park when idle, and the caller running chunk 0 inline while the
//! workers chew the rest. Dedicated threads only — no async, nothing
//! parked while holding anything.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

pub struct SendPtr<X>(pub *mut X);
unsafe impl<X> Send for SendPtr<X> {}
unsafe impl<X> Sync for SendPtr<X> {}

struct Task {
    range: (usize, usize),
    ctx: SendPtr<()>,
    run: fn(SendPtr<()>, usize, usize),
}

struct Worker {
    /// lock-free job slot: null = empty
    slot: AtomicPtr<Task>,
    /// slow-path parking (idle past the spin window)
    parked: Mutex<bool>,
    wake: Condvar,
}

impl Worker {
    fn new() -> Arc<Worker> {
        Arc::new(Worker {
            slot: AtomicPtr::new(std::ptr::null_mut()),
            parked: Mutex::new(false),
            wake: Condvar::new(),
        })
    }
}

pub struct Pool {
    workers: Vec<Arc<Worker>>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

/// The serving thread count we tune for (matches the lane's -t 10).
pub const SERVE_THREADS: usize = 10;

pub fn threads() -> usize {
    pool().workers.len()
}

pub fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .min(SERVE_THREADS)
            .saturating_sub(1) // caller participates
            .max(1);
        Pool::new(n)
    })
}

const SPIN: usize = 20_000;

/// The performance cores (GB10: 3.9GHz X925 clusters 5-9/15-19; the
/// lane's repro pins the same set). Empty on non-Linux/aarch64.
#[cfg(target_os = "linux")]
fn perf_cores() -> Vec<usize> {
    let mut cores = vec![];
    for i in 0..32 {
        let p = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq");
        if let Ok(f) = std::fs::read_to_string(&p) {
            if f.trim().parse::<u64>().unwrap_or(0) >= 3_900_000 {
                cores.push(i);
            }
        }
    }
    cores
}

#[cfg(target_os = "linux")]
fn pin_to_set(cores: &[usize]) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cores {
            libc::CPU_SET(c, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

impl Pool {
    fn new(n: usize) -> Pool {
        #[cfg(target_os = "linux")]
        let cores = {
            let c = perf_cores();
            if c.len() >= n {
                c
            } else {
                vec![]
            }
        };
        // Pin the CALLER to the full P-set BEFORE spawning: every
        // worker inherits the mask, so the whole pool stays on the
        // 3.9GHz clusters without per-core isolation (which the
        // scheduler punishes with 100x dispatch latency).
        #[cfg(target_os = "linux")]
        if !cores.is_empty() && std::env::var("TICK_PIN").is_ok() {
            pin_to_set(&cores);
        }
        let workers: Vec<Arc<Worker>> = (0..n).map(|_| Worker::new()).collect();
        for w in &workers {
            let w = Arc::clone(w);
            std::thread::Builder::new()
                .name("tick-par".into())
                .spawn(move || loop {
                    // fast path: spin on the slot; steal via swap
                    let mut taken: *mut Task = std::ptr::null_mut();
                    for _ in 0..SPIN {
                        taken = w.slot.swap(std::ptr::null_mut(), Ordering::AcqRel);
                        if !taken.is_null() {
                            break;
                        }
                        std::hint::spin_loop();
                    }
                    if taken.is_null() {
                        // park until a post wakes us
                        let mut parked = w.parked.lock().unwrap();
                        while taken.is_null() {
                            taken = w.slot.swap(std::ptr::null_mut(), Ordering::AcqRel);
                            if !taken.is_null() {
                                break;
                            }
                            *parked = true;
                            parked = w.wake.wait(parked).unwrap();
                        }
                    }
                    let task: Box<Task> = unsafe { Box::from_raw(taken) };
                    (task.run)(task.ctx, task.range.0, task.range.1);
                })
                .expect("spawn tick-par worker");
        }
        Pool { workers }
    }
}

fn post(w: &Worker, t: Task) {
    let p = Box::into_raw(Box::new(t));
    let old = w.slot.swap(p, Ordering::AcqRel);
    debug_assert!(old.is_null(), "overlapped post");
    let mut parked = w.parked.lock().unwrap();
    if *parked {
        *parked = false;
        drop(parked);
        w.wake.notify_all();
    }
}

/// A join's shared completion counter (lock-free; the caller spins).
struct Join {
    done: AtomicUsize,
}

impl Join {
    fn hit(&self) {
        self.done.fetch_add(1, Ordering::Release);
    }
    fn spin_until(&self, total: usize) {
        while self.done.load(Ordering::Acquire) < total {
            std::hint::spin_loop();
        }
    }
}

/// Run `f(start, end)` over ranges covering 0..total. Chunk 0 runs on
/// the CALLER thread; the rest round-robin over workers. Blocks until
/// every chunk returns.
pub fn for_chunks<F: Fn(usize, usize) + Sync>(total: usize, min_chunk: usize, f: F) {
    let p = pool();
    let n = total;
    if p.workers.is_empty() || n <= min_chunk {
        f(0, n);
        return;
    }
    let nw = p.workers.len();
    let nchunks = nw + 1;
    let per = n.div_ceil(nchunks);
    let nworkers_used = n.saturating_sub(1).div_ceil(per);
    let join = Arc::new(Join {
        done: AtomicUsize::new(0),
    });
    let mut posted = 0usize;
    let mut c = per;
    let mut i = 0usize;
    while c < n && posted < nworkers_used {
        let e = (c + per).min(n);
        let wrap = ChunkWrap { f: &f, join: &join };
        let ctx = SendPtr(&wrap as *const ChunkWrap<F> as *mut ChunkWrap<F>);
        let run = |ctx: SendPtr<()>, s: usize, e: usize| {
            let w: &ChunkWrap<F> = unsafe { &*(ctx.0 as *const ChunkWrap<F>) };
            (w.f)(s, e);
            w.join.hit();
        };
        post(
            &p.workers[i % nw],
            Task {
                range: (c, e),
                ctx: SendPtr(ctx.0 as *mut ()),
                run,
            },
        );
        i += 1;
        posted += 1;
        c = e;
    }
    f(0, per.min(n));
    join.spin_until(posted);
}

struct ChunkWrap<'a, F: ?Sized> {
    f: &'a F,
    join: &'a Arc<Join>,
}

/// Row-parallel apply: out\[i\] = f(i), chunks of `out` written by
/// distinct workers (disjoint slices — result equals the serial loop).
pub fn par_rows<T: Send, F: Fn(usize) -> T + Sync>(out: &mut [T], f: F) {
    let n = out.len();
    let p = pool();
    if p.workers.is_empty() || n < 256 {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = f(i);
        }
        return;
    }
    let nw = p.workers.len();
    let nchunks = nw + 1;
    let per = n.div_ceil(nchunks);
    let nworkers_used = n.saturating_sub(1).div_ceil(per);
    let join = Arc::new(Join {
        done: AtomicUsize::new(0),
    });
    let base = out.as_mut_ptr();
    let mut posted = 0usize;
    let mut c = per;
    let mut i = 0usize;
    while c < n && posted < nworkers_used {
        let e = (c + per).min(n);
        let wrap = RowsWrap {
            f: &f,
            base,
            join: &join,
            _m: std::marker::PhantomData,
        };
        let ctx = SendPtr(&wrap as *const RowsWrap<T, F> as *mut RowsWrap<T, F>);
        let run = |ctx: SendPtr<()>, s: usize, e: usize| {
            let w: &RowsWrap<T, F> = unsafe { &*(ctx.0 as *const RowsWrap<T, F>) };
            for idx in s..e {
                unsafe { *w.base.add(idx) = (w.f)(idx) };
            }
            w.join.hit();
        };
        post(
            &p.workers[i % nw],
            Task {
                range: (c, e),
                ctx: SendPtr(ctx.0 as *mut ()),
                run,
            },
        );
        i += 1;
        posted += 1;
        c = e;
    }
    for idx in 0..per.min(n) {
        unsafe { *base.add(idx) = f(idx) };
    }
    join.spin_until(posted);
}

struct RowsWrap<'a, T: Send, F: ?Sized> {
    f: &'a F,
    base: *mut T,
    join: &'a Arc<Join>,
    _m: std::marker::PhantomData<&'a T>,
}
