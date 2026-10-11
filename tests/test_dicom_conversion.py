import importlib.util
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

spec = importlib.util.spec_from_file_location('convert_dicom', Path(__file__).resolve().parents[1]/'scripts'/'convert_dicom.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def dicom(path, pixels, photometric):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype('<u2').tobytes()
    ds.save_as(path, enforce_file_format=True)


@pytest.mark.parametrize('photometric', ['MONOCHROME1','MONOCHROME2'])
def test_real_dicom_decode_and_polarity(tmp_path, photometric):
    source, output = tmp_path/'dicom', tmp_path/'png'
    source.mkdir()
    pixels = np.array([[0,100],[200,300]],dtype=np.uint16)
    dicom(source/'1.dcm',pixels,photometric)
    assert converter.convert_directory(source,output,2) == (1,0)
    result = np.asarray(Image.open(output/'1.png'))
    expected = np.array([[0,85],[170,255]],dtype=np.uint8)
    if photometric == 'MONOCHROME1': expected = 255-expected
    assert np.array_equal(result,expected)
    assert converter.convert_directory(source,output,2) == (0,1)


def test_constant_and_invalid_pixels():
    assert not converter.normalize_pixels(np.ones((2,2)), 'MONOCHROME2').any()
    with pytest.raises(ValueError): converter.normalize_pixels(np.ones((2,2,2)),'MONOCHROME2')
    with pytest.raises(ValueError): converter.normalize_pixels(np.full((2,2),np.nan),'MONOCHROME2')


def test_conversion_path_collisions_rejected(tmp_path):
    source = tmp_path/'input'
    source.mkdir()
    (source/'a.dcm').touch()
    (source/'a.dicom').touch()
    with pytest.raises(ValueError,match='same PNG'): converter.convert_directory(source,tmp_path/'out',2)
    with pytest.raises(ValueError,match='outside'): converter.convert_directory(source,source/'out',2)
