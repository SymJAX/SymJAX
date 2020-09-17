import io
import os
import time
from .utils import download_dataset
import zipfile

import numpy as np
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm
import pretty_midi
import soundfile as sf

_urls = {
    "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip": "groove-v1.0.0.zip"
}


def load(path=None):
    """
    The Groove MIDI Dataset (GMD) is composed of 13.6 hours of aligned MIDI and (synthesized) audio of human-performed, tempo-aligned expressive drumming. The dataset contains 1,150 MIDI files and over 22,000 measures of drumming.
    Size: 4.76GB

    License
    -------

    Creative Commons License

    The dataset is made available by Google LLC under a Creative Commons Attribution 4.0 International (CC BY 4.0) License.

    Dataset
    -------

    Update: If you’re looking for a dataset suitable for drum transcription or other audio-focused applications, see our Expanded Groove MIDI Dataset.

    To enable a wide range of experiments and encourage comparisons between methods on the same data, we created a new dataset of drum performances recorded in MIDI format. We hired professional drummers and asked them to perform in multiple styles to a click track on a Roland TD-11 electronic drum kit. We also recorded the aligned, high-quality synthesized audio from the TD-11 and include it in the release.

    The Groove MIDI Dataset (GMD), has several attributes that distinguish it from existing ones:

    The dataset contains about 13.6 hours, 1,150 MIDI files, and over 22,000 measures of drumming.
    Each performance was played along with a metronome set at a specific tempo by the drummer.
    The data includes performances by a total of 10 drummers, with more than 80% of duration coming from hired professionals. The professionals were able to improvise in a wide range of styles, resulting in a diverse dataset.
    The drummers were instructed to play a mix of long sequences (several minutes of continuous playing) and short beats and fills.
    Each performance is annotated with a genre (provided by the drummer), tempo, and anonymized drummer ID.
    Most of the performances are in 4/4 time, with a few examples from other time signatures.
    Four drummers were asked to record the same set of 10 beats in their own style. These are included in the test set split, labeled eval-session/groove1-10.
    In addition to the MIDI recordings that are the primary source of data for the experiments in this work, we captured the synthesized audio outputs of the drum set and aligned them to within 2ms of the corresponding MIDI files.

    A train/validation/test split configuration is provided for easier comparison of model accuracy on various tasks.
    Split   Beats   Fills   Measures (approx.)  Hits    Duration (minutes)
    Train   378     519     17752   357618  648.5
    Validation  48  76  2269    44044   82.2
    Test    77  52  2193    43832   84.3
    Total   503     647     22214   445494  815.0


    For more information about how the dataset was created and several applications of it, please see the paper where it was introduced: Learning to Groove with Inverse Sequence Transformations.

    For an example application of the dataset, see our blog post on GrooVAE.
    MIDI Data
    Format

    The Roland TD-11 splits the recorded data into separate tracks: one for meta-messages (tempo, time signature, key signature), one for control changes (hi-hat pedal position), and one for notes. The control changes are set on channel 0 and the notes on channel 9 (the canonical drum channel). To simplify processing of this data, we made two adustments to the raw MIDI files before distributing:

        We merged all messages (meta, control change, and note) to a single track.
        We set all messages to channel 9 (10 if 1-indexed).

    Drum Mapping
    ------------

    The Roland TD-11 used to record the performances in MIDI uses some pitch values that differ from the General MIDI (GM) Specifications. Below we show how the Roland mapping compares to GM. Please take note of these discrepancies during playback and training. The final column shows the simplified mapping we used in our paper.
    Pitch   Roland Mapping  GM Mapping  Paper Mapping   Frequency
    36  Kick    Bass Drum 1     Bass (36)   88067
    38  Snare (Head)    Acoustic Snare  Snare (38)  102787
    40  Snare (Rim)     Electric Snare  Snare (38)  22262
    37  Snare X-Stick   Side Stick  Snare (38)  9696
    48  Tom 1   Hi-Mid Tom  High Tom (50)   13145
    50  Tom 1 (Rim)     High Tom    High Tom (50)   1561
    45  Tom 2   Low Tom     Low-Mid Tom (47)    3935
    47  Tom 2 (Rim)     Low-Mid Tom     Low-Mid Tom (47)    1322
    43  Tom 3 (Head)    High Floor Tom  High Floor Tom (43)     11260
    58  Tom 3 (Rim)     Vibraslap   High Floor Tom (43)     1003
    46  HH Open (Bow)   Open Hi-Hat     Open Hi-Hat (46)    3905
    26  HH Open (Edge)  N/A     Open Hi-Hat (46)    10243
    42  HH Closed (Bow)     Closed Hi-Hat   Closed Hi-Hat (42)  31691
    22  HH Closed (Edge)    N/A     Closed Hi-Hat (42)  34764
    44  HH Pedal    Pedal Hi-Hat    Closed Hi-Hat (42)  52343
    49  Crash 1 (Bow)   Crash Cymbal 1  Crash Cymbal (49)   720
    55  Crash 1 (Edge)  Splash Cymbal   Crash Cymbal (49)   5567
    57  Crash 2 (Bow)   Crash Cymbal 2  Crash Cymbal (49)   1832
    52  Crash 2 (Edge)  Chinese Cymbal  Crash Cymbal (49)   1046
    51  Ride (Bow)  Ride Cymbal 1   Ride Cymbal (51)    43847
    59  Ride (Edge)     Ride Cymbal 2   Ride Cymbal (51)    2220
    53  Ride (Bell)     Ride Bell   Ride Cymbal (51)    5567
    Control Changes

    The TD-11 also records control changes specifying the position of the hi-hat pedal on each hit. We have preserved this information under control 4.

    How to Cite
    -----------

    If you use the Groove MIDI Dataset in your work, please cite the paper where it was introduced:

    Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, and David Bamman.
    "Learning to Groove with Inverse Sequence Transformations."
      International Conference on Machine Learning (ICML), 2019.

    You can also use the following BibTeX entry:

    @inproceedings{groove2019,
        Author = {Jon Gillick and Adam Roberts and Jesse Engel and Douglas Eck and David Bamman},
        Title = {Learning to Groove with Inverse Sequence Transformations},
        Booktitle = {International Conference on Machine Learning (ICML)},
        Year = {2019},
    }

    Acknowledgements
    ----------------

    We’d like to thank the following primary contributors to the dataset:

        Dillon Vado (of Never Weather)
        Jonathan Fishman (of Phish)
        Michaelle Goerlitz (of Wild Mango)
        Nick Woodbury (of SF Contemporary Music Players)
        Randy Schwartz (of El Duo)

    Additional drumming provided by: Jon Gillick, Mikey Steczo, Sam Berman, and Sam Hancock.

    """
    if path is None:
        path = os.environ["DATASET_PATH"]
    download_dataset(path, "groove_MIDI", _urls)

    t0 = time.time()

    # load wavs
    f = zipfile.ZipFile(os.path.join(path, "groove_MIDI", "groove-v1.0.0.zip"))

    columns = "drummer,session,id,style,bpm,beat_type,time_signature,midi_filename,audio_filename,duration,split".split(
        ","
    )
    info_file = f.open("groove/info.csv")
    infos = np.loadtxt(info_file, delimiter=",", dtype=str)

    data = [[] for i in range(len(columns))]

    indices = list(range(7)) + [9, 10]

    for row in tqdm(infos[1:], ascii=True):

        try:
            wav = f.read("groove/" + row[8])
            byt = io.BytesIO(wav)
            data[8].append(sf.read(byt)[0].astype("float32"))
        except RuntimeError:
            print("...skipping ", row[8])
            continue

        for column in indices:
            data[column].append(row[column])

        data[7].append(pretty_midi.PrettyMIDI(io.BytesIO(f.read("groove/" + row[7]))))

    data = {col: data for col, data in zip(columns, data)}

    return data
