# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and Open Climate Fix.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This dataset consists of HRV channel imagery from the EUMETSAT SEVIRI RSS service covering the UK from 2020-2021"""
import pandas as pd
import xarray
import zarr
import gcsfs
import datasets


_CITATION = """\
@InProceedings{eumetsat:ocf_uk_hrv,
title = {EUMETSAT SEVIRI RSS UK HRV},
author={EUMETSAT, with preparation by Open Climate Fix
},
year={2022}
}
"""

_DESCRIPTION = """\
The EUMETSAT Spinning Enhanced Visible and InfraRed Imager (SEVIRI) rapid scanning service (RSS) takes an image of the northern third of the Meteosat disc every five minutes (see the EUMETSAT website for more information on SEVIRI RSS ). The original EUMETSAT dataset  contains data from 2008 to the present day from 12 channels, and for a wide geographical extent covering North Africa, Saudi Arabia, all of Europe, and Western Russia. In contrast, this dataset on Google Cloud is a small subset of the entire SEVIRI RSS dataset: This Google Cloud dataset is from a single channel: the "high resolution visible" (HRV) channel; and contains data from January 2020 to November 2021. The geographical extent of this dataset on Google Cloud is a small subset of the total SEVIRI RSS extent: This Google Cloud dataset includes data over the United Kingdom and over North Western Europe.

This dataset is slightly transformed: It does not contain the original numerical values. 

The original data is copyright EUMETSAT. EUMETSAT has given permission to redistribute this transformed data. The data was transformed by Open Climate Fix using satip.

This public dataset is hosted in Google Cloud Storage and available free to use. 
"""

_HOMEPAGE = "https://console.cloud.google.com/marketplace/product/bigquery-public-data/eumetsat-seviri-rss-hrv-uk?project=tactile-acrobat-249716"

_LICENSE = "Cite EUMETSAT as the data source. This data is redistributed with permission from EUMETSAT under the terms of the EUMETSAT Data Policy for SEVIRI data with a latency of >3 hours . This redistributed dataset is released under the CC BY 4.0 open data license & is provided \"AS IS\" without any warranty, express or implied, from Google. Google disclaims all liability for any damages, direct or indirect, resulting from the use of the dataset."

_URL = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"


class EumetsatUkHrvDataset(datasets.GeneratorBasedBuilder):
    """This dataset consists of the HRV channel from the EUMETSAT SEVIRI RSS service covering the UK from 2020 to 2021."""

    VERSION = datasets.Version("1.2.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="uk", version=VERSION, description="This part of the dataset covers the UK"),
        datasets.BuilderConfig(name="uk_osgb", version=VERSION, description="This part of the dataset covers the UK in OSGB coordinates"),
        datasets.BuilderConfig(name="uk_video", version=VERSION, description="This dataset is for video prediction")
    ]

    DEFAULT_CONFIG_NAME = "uk_osgb"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if self.config.name == "uk":
            features = datasets.Features(
                {
                    "timestamp": datasets.Value("time64[ns]"),
                    "image": datasets.Array2D(shape=(891,1843), dtype="int16"),
                    "x_coordinates": datasets.Sequence(datasets.Value("float64")),
                    "y_coordinates": datasets.Sequence(datasets.Value("float64"))
                }
            )
        elif self.config.name == "uk_osgb":
            features = datasets.Features(
                {
                    "timestamp": datasets.Value("time64[ns]"),
                    "image": datasets.Array2D(shape=(891,1843), dtype="int16"),
                    "x_coordinates": datasets.Array2D(shape=(891,1843), dtype="float64"),
                    "y_coordinates": datasets.Array2D(shape=(891,1843), dtype="float64")
                }
            )
        else:
            features = datasets.Features(
                {
                    "timestamps":datasets.Sequence(datasets.Value("time64[ns]")),
                    "video": datasets.Array3D(shape=(36,891,1843), dtype="int16"),
                    "x_coordinates": datasets.Sequence(datasets.Value("float64")),
                    "y_coordinates": datasets.Sequence(datasets.Value("float64"))
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        #data_dir = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": _URL,
                    "time_range": slice("2020-01-01", "2020-12-31"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": _URL,
                    "time_range": slice("2021-01-01", "2021-12-31"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, time_range, split):
        sat_data = xarray.open_dataset(filepath, engine="zarr", chunks='auto')
        sat_data = sat_data.sel(time=time_range)
        if self.config.name == "uk_video":
            last_chunk_time = sat_data.time.values[0] - pd.Timedelta("3 hours")
            for key, timestamp in enumerate(sat_data.time.values):
                if timestamp >= last_chunk_time + pd.Timedelta("3 hours"):
                    # Get current time and go backwards an hour and forward 2 hours
                    start_time = timestamp - pd.Timedelta("55 minutes")
                    end_time = timestamp + pd.Timedelta("2 hours")
                    entry = sat_data.sel(time=slice(start_time, end_time))
                    # Only want to keep ones that have the correct length
                    if len(entry.time.values) == 36:
                        last_chunk_time = timestamp
                        yield key, {
                            "timestamps": entry.time.values,
                            "x_coordinates": entry.x.values,
                            "y_coordinates": entry.y.values,
                            "video": entry.values,
                        }
        else:
            for key, timestamp in enumerate(sat_data.time.values):
                if self.config.name == "uk":
                    entry = sat_data.sel(time=timestamp)
                    yield key, {
                        "timestamp": entry.time.values,
                        "x_coordinates": entry.x.values,
                        "y_coordinates": entry.y.values,
                        "image": entry.values,
                    }
                elif self.config.name == "uk_osgb":
                    entry = sat_data.sel(time=timestamp)
                    yield key, {
                        "timestamp": entry.time.values,
                        "x_coordinates": entry.x_osgb.values,
                        "y_coordinates": entry.y_osgb.values,
                        "image": entry.values,
                    }
                
# Load the dataset
builder = EumetsatUkHrvDataset()
builder.download_and_prepare()
ds = builder.as_dataset(split="train")
print(ds)