"""Functional tests with generated inputs only."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from scipy.fft import idct
from scec import SCEC, save_model, load_model
from scec.training import train_model
from scec.quantization import symbols, restore_doubled
from scec.peak_guard import PeakGuard
from examples.synthetic_roundtrip import synthetic_blocks


class CodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, cls.history = train_model(synthetic_blocks(4,128,91), synthetic_blocks(2,128,92),
                                            [80.,240.], bands=4, epochs=1, batch_size=2)

    def test_variable_length_roundtrips(self):
        encoder = SCEC(copy.deepcopy(self.model))
        decoder = SCEC(copy.deepcopy(self.model))
        for n in [1,7,8,9,63,128,257,1025]:
            x = synthetic_blocks(1,n,100+n)[0]
            for step in [80.,240.]:
                payload = encoder.encode(x,step)
                padded = np.pad(x,(0,(-n)%8),mode='reflect' if n>1 else 'edge')
                q,c = symbols(padded,step,'e8')
                expected = idct(restore_doubled(c,q)*.5*step,norm='ortho')[:n]
                np.testing.assert_array_equal(decoder.decode(payload),expected)

    def test_independent_process_and_model_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'model'
            save_model(self.model,folder)
            loaded,_ = load_model(folder)
            x = synthetic_blocks(1,257,101)[0]
            payload = SCEC(self.model).encode(x,80.)
            self.assertEqual(payload,SCEC(loaded).encode(x,80.))
            path = Path(temporary)/'synthetic.scec';path.write_bytes(payload)
            expected = SCEC(loaded).decode(payload)
            np.save(Path(temporary)/'expected.npy',expected)
            script = ('from pathlib import Path; import numpy as np; from scec import SCEC,load_model; '
                      'import sys; p=Path(sys.argv[1]); m,_=load_model(p/"model"); '
                      'np.testing.assert_array_equal(SCEC(m).decode((p/"synthetic.scec").read_bytes()),np.load(p/"expected.npy"))')
            subprocess.run([sys.executable,'-c',script,temporary],check=True)

    def test_protection_and_validation(self):
        codec = SCEC(copy.deepcopy(self.model))
        guard = PeakGuard(codec)
        x = np.array([-32768,32767,0,-4,20,100,-900],dtype=np.int16)
        for threshold in [0.,10.,500.]:
            y = guard.decode(guard.encode(x,240.,threshold))
            self.assertLessEqual(np.max(np.abs(x.astype(float)-y)),threshold)
        for invalid in [np.array([]),np.array([np.nan]),np.array([1.2]),np.array([40000]),np.zeros((2,2))]:
            with self.assertRaises(ValueError):codec.encode(invalid,80.)
        with self.assertRaises(ValueError):codec.decode(b'bad')
        payload=codec.encode(np.zeros(16,dtype=np.int16),80.)
        with self.assertRaises(ValueError):codec.decode(payload[:-1])
        with self.assertRaises(ValueError):codec.encode(np.zeros(8,dtype=np.int16),100.)


if __name__ == '__main__':
    unittest.main()
