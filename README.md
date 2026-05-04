# mosox benchmark

Comparing [mosox](https://github.com/carderne/mosox) to GMPL and [linopy](https://github.com/PyPSA/linopy).

I don't claim these benchmarks to be authoritative.
They're based on an LLM-generated linopy implementation of [Osemosys](https://github.com/OSeMOSYS/OSeMOSYS) with the dummy "Atlantis" model.

I made a decent effort to make the comparison fair, but I don't understand Osemosys _or_ linopy well enough to be very confident in that.

See further caveats below.

## Setup
In order to run this benchmark, you will need:
- Python (preferably [uv](https://docs.astral.sh/uv/getting-started/installation/))
- [mosox](https://rdrn.me/mosox/)

Then from this directory, run:
```bash
uv sync
```

And then run:
```bash
uv run bench.py --help
```

## Benchmarks

Notes:
- Osemosys "Atlantis" model. From [here](http://www.osemosys.org/uploads/1/8/5/0/18504136/atlantis_bau.txt) (retrived December 2025, link now dead).
- Manually converted to linopy with GPT-5.5.
- Validated to produce the same results.
- Not validated to be optimal linopy code (check [./lino](./lino)).
- Used the standard Osemosys model, not the fast version.
- Matrix column below is the time to output an MPS file.
- Matrix + solve is the full time from input to solution.
- Average of 9 runs (interleaved).
- For the full details on how the benchmark is setup, check [./bench.py](./bench.py).
- Run on a Macbook Air with M1 and 24 GB of memory.

### Small Model

```bash
uv run bench.py 9 --size sm
```

| Tool   | Matrix (s)| Matrix + solve (s)|
| ------ | --------: | -----------------:|
| GMPL   |      2.44 |             11.67 |
| Linopy |      3.89 |              5.41 |
| Mosox  |      0.59 |              0.89 |

For the full run, Linopy is ~2x faster than GMPL.

Mosox is a further ~6x faster than linopy.

### Large Model

Same model, but with some sets extended (details below).

```bash
uv run bench.py 9 --size lg
```

| Tool   | Matrix (s)| Matrix + solve (s)|
| ------ | ---------:| -----------------:|
| GMPL   |     23.81 |             62.15 |
| Linopy |     10.78 |             17.70 |
| Mosox  |      5.12 |              6.88 |

For the full run, Linopy is ~3.5x faster than GMPL.

Mosox is a further ~2.5x faster than linopy.

#### Small vs large comparison

These are the sets that are different between the two:

```yaml
 - REGION
     - sm: Atlantis_00A
     - lg: Atlantis_00A … Atlantis_00H (8 regions)
 - MODE_OF_OPERATION
     - sm: 1 2
     - lg: 1 2 3
 - EMISSION
     - sm: CO2 NOx
     - lg: CO2 NOx SO2 CH4 PM25
 - STORAGE
     - sm: DAM
     - lg: DAM BATTERY HYDROGEN
```
