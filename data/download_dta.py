import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data

RESOURCES = [
    DownloadableFile( # dnli data
        '1WtbXCv3vPB5ql6w0FVDmAEMmWadbrCuG',
        'dialogue_nli.zip',
        '1164b0d9a0a1a6006891a6d4435a6a813464bc9b1e2f1ec5ce28c47267ad5e42',
        from_google=True,
    ),
    DownloadableFile( # personachat data
        'http://parl.ai/downloads/personachat/personachat.tgz',
        'personachat.tgz',
        '507cf8641d333240654798870ea584d854ab5261071c5e3521c20d8fa41d5622',
    )
]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'raw_data')
    # define version if any
    version = '1.0'

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)

build({'datapath': './'})