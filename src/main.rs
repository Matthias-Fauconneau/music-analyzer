#![feature(exact_size_is_empty, impl_trait_in_assoc_type)]#![feature(slice_from_ptr_range)]/*shader*/#![allow(non_snake_case,mixed_script_confusables)]
mod audio;
pub trait Decoder<Packet> { type Buffer<'t> where Self: 't; fn decode(&mut self, _: &Packet) -> Self::Buffer<'_>; }
pub trait SplitConvert<T> { type Channel<'t>: ExactSizeIterator<Item=T>+'t where Self: 't; fn split_convert<'t>(&'t self) -> [Self::Channel<'t>; 2]; }
#[cfg(feature="resample")] mod resampler;
fn default<T: Default>() -> T { Default::default() }
type Result<T = (), E = Box<dyn std::error::Error>>  = std::result::Result<T, E>;

use symphonia::core::{formats, codecs};

fn open(path: &std::path::Path) -> Result<(Box<dyn formats::FormatReader>, Box<dyn codecs::Decoder>)> {
	let file = symphonia::default::get_probe().format(&symphonia::core::probe::Hint::new().with_extension(path.extension().ok_or("")?.to_str().unwrap()),
		symphonia::core::io::MediaSourceStream::new(Box::new(std::fs::File::open(path)?), default()), &default(), &default())?;
	let container = file.format;
	let decoder = symphonia::default::get_codecs().make(&container.default_track().unwrap().codec_params, &default())?;
	Ok((container, decoder))
}

use audio::{PCM, Write as _};
#[derive(Default)] pub struct Player {
	output: Vec<PCM>,
}
impl Player {
	fn new(output: &[&str]) -> Self { Self{output: Vec::from_iter(output.into_iter().map(|path| PCM::new(path, 48000).unwrap()))} }
}

use {std::sync::Arc, parking_lot::{Mutex, MutexGuard}};
#[derive(Default,Clone)] struct Arch<T>(Arc<Mutex<T>>);
impl<T> Arch<T> {
    pub fn new(inner: T) -> Self { Self(std::sync::Arc::new(Mutex::new(inner))) }
    pub fn lock(&self) -> MutexGuard<'_, T> { self.0.lock() }
    pub fn clone(&self) -> Self { Self(self.0.clone()) }
}
unsafe impl<T> Send for Arch<T> {}
unsafe impl<T> Sync for Arch<T> {}

#[cfg(feature="resample")] type Resampler = resampler::MultiResampler;
#[cfg(not(feature="resample"))] struct Resampler;
#[cfg(not(feature="resample"))] impl Resampler {
	fn new(input: u32, output: u32) -> Option<Self> { assert_eq!(input, output); None }
	pub fn resample<Packets: Iterator, D: Decoder<Packets::Item>>(&mut self, _: &mut Packets, _: &mut D) -> Option<[std::slice::Iter<'_, f32>; 2]> where for<'t> D::Buffer<'t>: SplitConvert<f32> { unimplemented!() }
}

use {std::borrow::Cow, symphonia::core::{formats::Packet, audio::{AudioBufferRef, AudioBuffer, Signal as _}, sample::{self, Sample, SampleFormat}, conv}};
trait Cast<'t, S:Sample> { fn cast(self) -> Cow<'t, AudioBuffer<S>>; }
impl<'t> Cast<'t, i32> for AudioBufferRef<'t> { fn cast(self) -> Cow<'t, AudioBuffer<i32>> { if let AudioBufferRef::S32(variant) = self  { variant } else { unreachable!() } } }
impl<'t> Cast<'t, f32> for AudioBufferRef<'t> { fn cast(self) -> Cow<'t, AudioBuffer<f32>> { if let AudioBufferRef::F32(variant) = self  { variant } else { unreachable!() } } }
impl<S:Sample, T:conv::FromSample<S>> SplitConvert<T> for std::borrow::Cow<'_, AudioBuffer<S>> {
	type Channel<'t> = impl ExactSizeIterator<Item=T>+'t where Self: 't;
	fn split_convert<'t>(&'t self) -> [Self::Channel<'t>; 2]  { [0,1].map(move |channel| self.chan(channel).iter().map(|&v| conv::FromSample::from_sample(v))) }
}

struct TypedDecoder<D,S>(D, std::marker::PhantomData<S>); // S type checks the audio buffer sample type
impl<S:Sample+'static> Decoder<Packet> for TypedDecoder<Box<dyn codecs::Decoder>, S> where for<'t> AudioBufferRef<'t>: Cast<'t, S> {
	type Buffer<'t> = Cow<'t, AudioBuffer<S>> where Self: 't;
	fn decode(&mut self, packet: &Packet) -> Self::Buffer<'_> { self.0.decode(packet).unwrap().cast() }
}

fn write <S: sample::Sample+'static, D, Output: std::ops::DerefMut<Target=[self::PCM; N]>, const N: usize>
	(resampler: &mut Option<Resampler>, ref mut packets: impl Iterator<Item=Packet>, decoder: D, ref mut output: impl FnMut() -> Output) -> audio::Result
	where TypedDecoder<D, S>: Decoder<Packet>,
	for <'t> <TypedDecoder<D, S> as Decoder<Packet>>::Buffer<'t>: SplitConvert<f32>,
	for <'t> <TypedDecoder<D, S> as Decoder<Packet>>::Buffer<'t>: SplitConvert<i16> {
	if let Some(resampler) = resampler.as_mut() {
		let mut decoder = TypedDecoder(decoder, std::marker::PhantomData/*S*/);
		while let Some([L, R]) = resampler.resample(packets, &mut decoder) {
			let f32_to_i16 = |s| f32::clamp(s*32768., -32768., 32767.) as i16;
			output.write(L.zip(R).map(|(L,R)| [L,R]).map(|[L,R]|[L,R].map(f32_to_i16)))?;
		}
	} else {
		let mut decoder = TypedDecoder(decoder, std::marker::PhantomData/*S*/);
		for ref packet in packets {
			let ref buffer = Decoder::decode(&mut decoder, packet);
			let [L, R] = SplitConvert::<i16>::split_convert(buffer);
			output.write(L.zip(R).map(|(L,R)| [L,R]))?;
		}
	}
	Ok(())
}

use ui::{size, int2, image::{Image, xy, rgba8}, Widget, EventContext, Event, vulkan, shader, run};
use vulkan::{Context, Commands, ImageView, PrimitiveTopology, image, WriteDescriptorSet, linear};
shader!{view}
impl<T:Widget> Widget for Arch<T> {
	fn paint(&mut self, context: &Context, commands: &mut Commands, target: Arc<ImageView>, size: size, offset: int2) -> Result { self.lock().paint(context, commands, target, size, offset) }
	fn event(&mut self, context: &Context, commands: &mut Commands, size: size, event_context: &mut EventContext, event: &Event) -> Result<bool> { self.lock().event(context, commands, size, event_context, event) }
}

impl Widget for Player {
	fn paint(&mut self, context: &Context, commands: &mut Commands, target: Arc<ImageView>, size: size, _: int2) -> Result {
		let pass = view::Pass::new(context, false, PrimitiveTopology::TriangleList)?;
		let image = image(context, commands, Image::from_xy(size, |xy{x,y}| rgba8{r: if x%2==0 { 0 } else { 0xFF }, g: if y%2==0 { 0 } else { 0xFF }, b: 0xFF, a: 0xFF}).as_ref())?;
		pass.begin_rendering(context, commands, target.clone(), None, true, &view::Uniforms::empty(), &[
			WriteDescriptorSet::image_view(0, ImageView::new_default(&image)?),
        WriteDescriptorSet::sampler(1, linear(context)),
		])?;
		unsafe{commands.draw(3, 1, 0, 0)}?;
		commands.end_rendering()?;
		Ok(())
	}
}

fn main() -> Result {
	let ref path = std::env::args().skip(1).next().map(std::path::PathBuf::from).unwrap();
	let [L,R] = {
		let (mut reader, mut decoder) = open(path)?;
		let mut packets = ;
		let mut decoder = TypedDecoder(decoder, std::marker::PhantomData::<i32>);
		let [mut Ls, mut Rs] = [const{Vec::new()}; 2];
		for ref packet in std::iter::from_fn(|| reader.next_packet().ok()) {
			let ref buffer = Decoder::decode(&mut decoder, packet);
			let [L, R] = SplitConvert::<f32>::split_convert(buffer);
			Ls.append(&mut Vec::from_iter(L));
			Rs.append(&mut Vec::from_iter(R));
		}
		[Ls, Rs]
	};
	/*let (mut reader, mut decoder) = open(path)?;
	const N: usize = 2;
	let player : Arch<Player> = Arch::new(Player::new(if N == 1 {&["/dev/snd/pcmC0D0p"]} else {&["/dev/snd/pcmC0D2p","/dev/snd/pcmC0D0p"]}));
	let ref mut resampler = Resampler::new(decoder.codec_params().sample_rate.unwrap(), player.lock().output[0].rate);
	use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
	let ref stop = AtomicBool::new(false);
	std::thread::scope(|s| {
		std::thread::Builder::new().spawn_scoped(s, {let player : Arch<Player> = Arch::clone(&player); move || {
			let mut packets = std::iter::from_fn(|| (!stop.load(Relaxed)).then(|| reader.next_packet().ok()).flatten());
			let sample_format = decoder.codec_params().sample_format.unwrap_or_else(|| match decoder.decode(&packets.next().unwrap()).unwrap() {
				AudioBufferRef::S32(_) => SampleFormat::S32,
				AudioBufferRef::F32(_) => SampleFormat::F32,
				_ => unimplemented!(),
			});
			let output = || MutexGuard::map(player.lock(), |unlocked_player| <&mut [PCM; N]>::try_from(unlocked_player.output.as_mut_slice()).unwrap());
			match sample_format {
				SampleFormat::S32 => write::<i32, _, _, N>(resampler, packets, decoder, output),
				SampleFormat::F32 => write::<f32, _, _, N>(resampler, packets, decoder, output),
				_ => unimplemented!(),
			}
		}})?;
		let r = run(&path.display().to_string(), Box::new(|_,_| Ok(Box::new(player))));
		stop.store(true, Relaxed);
		println!("stop");
		r
	})*/
}

