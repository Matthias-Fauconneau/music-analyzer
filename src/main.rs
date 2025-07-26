#![feature(exact_size_is_empty, impl_trait_in_assoc_type)]
#![feature(slice_from_ptr_range)] /*shader*/
#![allow(non_snake_case, mixed_script_confusables)]
mod audio;
pub trait Decoder<Packet> {
	type Buffer<'t> where Self: 't;
	fn decode(&mut self, _: &Packet) -> Self::Buffer<'_>;
}
pub trait SplitConvert<T> {
	type Channel<'t>: ExactSizeIterator<Item = T> + 't where Self: 't;
	fn split_convert<'t>(&'t self) -> [Self::Channel<'t>; 2];
}
#[cfg(feature = "resample")]
mod resampler;
fn default<T: Default>() -> T { Default::default() }
type Result<T = (), E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

use symphonia::core::{codecs, formats};

fn open(path: &std::path::Path) -> Result<(Box<dyn formats::FormatReader>, Box<dyn codecs::Decoder>)> {
	let file = symphonia::default::get_probe().format(
		&symphonia::core::probe::Hint::new().with_extension(path.extension().ok_or("")?.to_str().unwrap()),
		symphonia::core::io::MediaSourceStream::new(Box::new(std::fs::File::open(path)?), default()),
		&default(), &default())?;
	let container = file.format;
	let decoder = symphonia::default::get_codecs().make(&container.default_track().unwrap().codec_params, &default())?;
	Ok((container, decoder))
}

use audio::{PCM, Write as _};
const K: usize = 256; // quarter tone bins
const N0: usize = 2048; // ~ 23Hz @48KHz
pub struct Player { 
	output: Vec<PCM>,
	image: Image<Box<[f32]>>,
	f: Box<[f32]>,
	t: usize,
	F: [(f32, f32); K],
}
impl Player {
	fn new(f: Box<[f32]>, output: &[&str]) -> Self { Self {
		output: Vec::from_iter(output.into_iter().map(|path| PCM::new(path, 48000).unwrap())),
		image: Image::uninitialized(xy{x: 3840, y: K as u32}),
		f,
		t: N0/2,
		F: [(0., 0.); K],
	} }
}

mod arch {
	use {parking_lot::{Mutex, MutexGuard, MappedMutexGuard}, std::sync::Arc};
	#[derive(Default, Clone)] pub struct Arch<T>(Arc<Mutex<T>>);
	impl<T> Arch<T> {
		pub fn new(inner: T) -> Self { Self(std::sync::Arc::new(Mutex::new(inner))) }
		pub fn lock(&self) -> MutexGuard<'_, T> { self.0.lock() }
		pub fn clone(&self) -> Self { Self(self.0.clone()) }
		pub fn map<U: ?Sized>(&self, f: impl FnOnce(&mut T)->&mut U) -> MappedMutexGuard<'_, U> { MutexGuard::map(self.lock(), f) }
	}
	unsafe impl<T> Send for Arch<T> {}
	unsafe impl<T> Sync for Arch<T> {}
}
use arch::Arch;

#[cfg(feature = "resample")] type Resampler = resampler::MultiResampler;
#[cfg(not(feature = "resample"))] struct Resampler;
#[cfg(not(feature = "resample"))] impl Resampler {
	fn new(input: u32, output: u32) -> Option<Self> { assert_eq!(input, output); None }
    pub fn resample<Packets: Iterator, D: Decoder<Packets::Item>>(&mut self, _: &mut Packets, _: &mut D) -> Option<[std::slice::Iter<'_, f32>; 2]>
    where for<'t> D::Buffer<'t>: SplitConvert<f32> { unimplemented!() }
}

use {std::borrow::Cow, symphonia::core::{audio::{AudioBuffer, AudioBufferRef, Signal as _}, conv, formats::Packet, sample::{self, Sample, SampleFormat}}};
trait Cast<'t, S: Sample> { fn cast(self) -> Cow<'t, AudioBuffer<S>>; }
impl<'t> Cast<'t, i32> for AudioBufferRef<'t> { fn cast(self) -> Cow<'t, AudioBuffer<i32>> { if let AudioBufferRef::S32(variant) = self { variant } else { unreachable!() } }}
impl<'t> Cast<'t, f32> for AudioBufferRef<'t> { fn cast(self) -> Cow<'t, AudioBuffer<f32>> { if let AudioBufferRef::F32(variant) = self { variant } else { unreachable!() } }}
impl<S: Sample, T: conv::FromSample<S>> SplitConvert<T> for std::borrow::Cow<'_, AudioBuffer<S>> {
	type Channel<'t> = impl ExactSizeIterator<Item = T> + 't where Self: 't;
	fn split_convert<'t>(&'t self) -> [Self::Channel<'t>; 2] { [0, 1].map(move |channel| self.chan(channel).iter().map(|&v| conv::FromSample::from_sample(v)) ) }
}

struct TypedDecoder<D, S>(D, std::marker::PhantomData<S>); // S type checks the audio buffer sample type
impl<S: Sample + 'static> Decoder<Packet> for TypedDecoder<Box<dyn codecs::Decoder>, S> where for<'t> AudioBufferRef<'t>: Cast<'t, S> {
	type Buffer<'t> = Cow<'t, AudioBuffer<S>> where Self: 't;
	fn decode(&mut self, packet: &Packet) -> Self::Buffer<'_> { self.0.decode(packet).unwrap().cast() }
}

fn write<S: sample::Sample + 'static, D, Output: std::ops::DerefMut<Target = [self::PCM; N]>, const N: usize>
	(resampler: &mut Option<Resampler>, ref mut packets: impl Iterator<Item = Packet>, decoder: D, ref mut output: impl FnMut() -> Output) -> audio::Result
	where TypedDecoder<D, S>: Decoder<Packet>, 
	for<'t> <TypedDecoder<D, S> as Decoder<Packet>>::Buffer<'t>: SplitConvert<f32>,
	for<'t> <TypedDecoder<D, S> as Decoder<Packet>>::Buffer<'t>: SplitConvert<i16> {
	if let Some(resampler) = resampler.as_mut() {
		let mut decoder = TypedDecoder(decoder, std::marker::PhantomData /*S*/);
		while let Some([L, R]) = resampler.resample(packets, &mut decoder) {
			let f32_to_i16 = |s| f32::clamp(s * 32768., -32768., 32767.) as i16;
			output.write( L.zip(R) .map(|(L, R)| [L, R]) .map(|[L, R]| [L, R].map(f32_to_i16)) )?;
		}
	} else {
		let mut decoder = TypedDecoder(decoder, std::marker::PhantomData /*S*/);
		for ref packet in packets {
			let ref buffer = Decoder::decode(&mut decoder, packet);
			let [L, R] = SplitConvert::<i16>::split_convert(buffer);
			output.write(L.zip(R).map(|(L, R)| [L, R]))?;
		}
	}
	Ok(())
}

use ui::{Event, EventContext, Widget, vector::{int2, size, max}, image::{Image, xy, rgb8, sRGB8_OETF12, oetf8_12}, vulkan, shader, run, new_trigger, trigger};
use vulkan::{Commands, Context, Arc, ImageView, PrimitiveTopology, WriteDescriptorSet, image, linear};
shader! {view}
impl<T: Widget> Widget for Arch<T> {
	fn paint(&mut self, context: &Context, commands: &mut Commands, target: Arc<ImageView>, size: size, offset: int2) -> Result {
		self.lock().paint(context, commands, target, size, offset) }
	fn event(&mut self, context: &Context, commands: &mut Commands, size: size, event_context: &mut EventContext, event: &Event) -> Result<bool> {
		self.lock().event(context, commands, size, event_context, event) }
}

impl Widget for Player {
	fn paint(&mut self, context: &Context, commands: &mut Commands, target: Arc<ImageView>, size: size, _: int2) -> Result {
		let pass = view::Pass::new(context, false, PrimitiveTopology::TriangleList)?;
		let Self{output, f, t, F, image} = self;
		let N: [u16; K] = std::array::from_fn(|k| (N0 as f64 / f64::exp2(k as f64 / 24.)) as u16);
		let Q: f64 = 1. / (f64::exp2(1. / 24.) - 1.);
		fn expi(x: f64) -> (f32, f32) { let (s, c) = f64::sin_cos(x); (c as f32, s as f32) }
		let R: [(f32, f32); K] = std::array::from_fn(|k| expi(2. * std::f64::consts::PI * Q / N[k] as f64));
		let r = expi(-2. * std::f64::consts::PI * Q);
		while *t < output[0].t {
			let x = (*t-N0/2) as u32 % image.size.x;
			for k in 0..K {
				let Nk = N[k] as usize;
				// /*Nk·*/ F[k] = R[k] * (F[k] + r * f[t + Nk / 2] - f[t - Nk / 2])
				let f =  {
					let F = F[k];
					let f = (f[*t - Nk / 2], f[*t + Nk / 2]);
					(F.0 + r.0 * f.1 - f.0 , F.1 + r.1 * f.1)
				};
				let R = R[k];
				F[k] = ((R.0*f.0 - R.1*f.1), R.0*f.1 + R.1*f.0);
				image[xy{x, y: k as u32}] = {let f = F[k]; f.0*f.0 + f.1*f.1};
			}
			*t += 1;
		}
		assert_eq!(image.size.x, size.x);
		let max = max(image.data.iter().copied()).unwrap();
		let ref oetf = sRGB8_OETF12;
		let image = self::image(context, commands, image.as_ref().map(|f| rgb8::from(oetf8_12(oetf, f/max)).into()).as_ref())?;
		pass.begin_rendering(context, commands, target.clone(), None, true, &view::Uniforms::empty(), &[
			WriteDescriptorSet::image_view(0, ImageView::new_default(&image)?),
			WriteDescriptorSet::sampler(1, linear(context)),
		])?;
		unsafe { commands.draw(3, 1, 0, 0) }?;
		commands.end_rendering()?;
		Ok(())
	}
	fn event(&mut self, _: &Context, _: &mut Commands, _: size, _: &mut EventContext, event: &Event) -> Result<bool> { Ok(if let Event::Idle|Event::Trigger = event {
		self.t < self.output[0].t
	} else { false }) }
}

fn main() -> Result {
	let ref path = std::env::args().skip(1).next().map(std::path::PathBuf::from).unwrap();
	let f = {
		let (mut reader, decoder) = open(path)?;
		let mut decoder = TypedDecoder(decoder, std::marker::PhantomData::<i32>);
		let mut f = Vec::with_capacity(0 /*FIXME*/);
		for ref packet in std::iter::from_fn(|| reader.next_packet().ok()) {
			let ref buffer = Decoder::decode(&mut decoder, packet);
			let [l, r] = SplitConvert::<f32>::split_convert(buffer);
			let frame = l.zip(r).map(|(l, r)| l + r);
			//f.extend_reserve(frame.len());
			f.extend(frame);
		}
		f.into_boxed_slice()
	};
	let player : Arch<Player> = Arch::new(Player::new(f, if true {&["/dev/snd/pcmC0D2p"]} else {&["/dev/snd/pcmC0D2p","/dev/snd/pcmC0D0p"]}));
	let (mut reader, mut decoder) = open(path)?;
    let ref mut resampler = Resampler::new(decoder.codec_params().sample_rate.unwrap(), player.lock().output[0].rate);
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    let ref stop = AtomicBool::new(false);
    let ref fd = new_trigger()?;
    std::thread::scope(|s| {
        std::thread::Builder::new().spawn_scoped(s, {let player : Arch<Player> = Arch::clone(&player); move || {
            let mut packets = std::iter::from_fn(|| (!stop.load(Relaxed)).then(|| reader.next_packet().ok()).flatten());
            let sample_format = decoder.codec_params().sample_format.unwrap_or_else(|| match decoder.decode(&packets.next().unwrap()).unwrap() {
                AudioBufferRef::S32(_) => SampleFormat::S32,
                AudioBufferRef::F32(_) => SampleFormat::F32,
                _ => unimplemented!(),
            });
            let output = || player.map(|player| { trigger(fd).unwrap(); <&mut [PCM; 1]>::try_from(player.output.as_mut_slice()).unwrap() });
            match sample_format {
                SampleFormat::S32 => write::<i32, _, _, 1>(resampler, packets, decoder, output),
                SampleFormat::F32 => write::<f32, _, _, 1>(resampler, packets, decoder, output),
                _ => unimplemented!(),
            }
        }})?;
        let r = run(fd, &path.display().to_string(), Box::new(|_,_| Ok(Box::new(player))));
        stop.store(true, Relaxed);
        r
    })
}
