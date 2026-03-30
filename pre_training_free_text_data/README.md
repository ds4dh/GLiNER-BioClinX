Please follow **Steps 0 to 3** to prepare your raw data (extract, clean, deduplicate, and create the initial split).

### Step 4a: Manage 12k dataset

We added `4a_manage_12k_dataset.py` to allow flexible updates to the final `balanced_12k.csv` without re-running the entire pipeline.

**Use case:**
We used this to replace the **Dutch (nl)** subset (switching from BioLord to Wikipedia) and add **Portuguese (pt)**.

**Usage:**
Place your new `nl` and `pt` CSVs in the deduplicated folder and run:

```bash
# This removes old 'nl' rows, adds the new 'nl' data,
# detects and adds 'pt' automatically, and reshuffles everything.
python 4a_manage_12k_dataset.py --replace nl
```
