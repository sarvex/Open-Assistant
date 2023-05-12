import gzip
import os
import shutil
import sys
from pathlib import Path

import numpy


def xor_uncompressed(dst, src_payload, src_base, block_size=4096):
    with open(src_payload, "rb") as fp_payload:
        fp_base = open(src_base, "rb")
        with open(dst, "wb") as fp:
            while True:
                buf1 = numpy.array(bytearray(fp_payload.read(block_size)), dtype=numpy.uint8)
                buf2 = numpy.array(bytearray(fp_base.read(block_size)), dtype=numpy.uint8)
                padding = len(buf1) - len(buf2)
                if padding > 0:
                    buf2 = numpy.pad(buf2, (0, padding), "constant", constant_values=(0,))
                if padding < 0:
                    buf2 = buf2[: len(buf1)]
                buf = numpy.bitwise_xor(buf1, buf2)
                fp.write(buf)
                if len(buf1) < block_size:
                    break
    fp_base.close()


def xor_encode(dst, src_payload, src_base, block_size=4096):
    with open(src_payload, "rb") as fp_payload:
        fp_base = open(src_base, "rb")
        with gzip.open(dst, "wb") as fp:
            while True:
                buf1 = numpy.array(bytearray(fp_payload.read(block_size)), dtype=numpy.uint8)
                buf2 = numpy.array(bytearray(fp_base.read(block_size)), dtype=numpy.uint8)
                padding = len(buf1) - len(buf2)
                if padding > 0:
                    buf2 = numpy.pad(buf2, (0, padding), "constant", constant_values=(0,))
                if padding < 0:
                    buf2 = buf2[: len(buf1)]
                buf = numpy.bitwise_xor(buf1, buf2)
                fp.write(buf)
                if len(buf1) < block_size:
                    break
    fp_base.close()


def xor_decode(dst, src_payload, src_base, block_size=4096):
    fp_payload = gzip.open(src_payload, "rb")
    with open(src_base, "rb") as fp_base:
        with open(dst, "wb") as fp:
            while True:
                buf1 = numpy.array(bytearray(fp_payload.read(block_size)), dtype=numpy.uint8)
                buf2 = numpy.array(bytearray(fp_base.read(block_size)), dtype=numpy.uint8)
                padding = len(buf1) - len(buf2)
                if padding > 0:
                    buf2 = numpy.pad(buf2, (0, padding), "constant", constant_values=(0,))
                if padding < 0:
                    buf2 = buf2[: len(buf1)]
                buf = numpy.bitwise_xor(buf1, buf2)
                fp.write(buf)
                if len(buf1) < block_size:
                    break
        fp_payload.close()


def xor_dir(dst, src_payload, src_base, decode=True, compress=True):
    if compress:
        xor = xor_decode if decode else xor_encode
    else:
        xor = xor_uncompressed
    Path(dst).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(src_payload) / "added_tokens.json", Path(dst) / "added_tokens.json")
    for path in os.listdir(src_payload):
        print(f"[*] Processing '{path}'")
        try:
            xor(f"{dst}/{path}", f"{src_payload}/{path}", f"{src_base}/{path}")
        except Exception:
            print(f"Exception when processing '{path}'")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: xor.py <DESTINATION> <PAYLOAD SOURCE> <LLAMA SOURCE> [--encode] [--compress]")
        exit()
    dst = sys.argv[1]
    src_payload = sys.argv[2]
    src_base = sys.argv[3]
    decode = True
    compress = False
    if len(sys.argv) > 4:
        for arg in sys.argv[4:]:
            if arg == "--compress":
                compress = True
            elif arg == "--encode":
                decode = False
    xor_dir(dst, src_payload, src_base, decode=decode, compress=compress)
