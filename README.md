# 🏹 archery_tools

Tools for fitting and evaluating archery sight marks.

## Local use

Requires Bash and Python 3. The Actions workflow uses Python 3.12.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Enter measured distances in meters and one sight mark per distance. For example:

```bash
bash ./fit_marks.sh --python python3 --distances 20,30,40 --marks 2.8,3.6,4.4 --mode polynomial --degree 2 --no-plot --start 20 --end 40 --step 10
```

The `polynomial` mode fits a polynomial of the selected degree. The `curve_fit` mode fits a model selected with `--model`. Both print the measured and evaluated marks; curve fitting also prints optimized parameters. By default, evaluations cover 1 to 50 meters in 1-meter steps, and a plot is saved in the current directory. Use `--no-plot` to skip the plot.

## GitHub Actions

To run a fit on GitHub, open the repository's **Actions** tab, select **Fit sight marks**, and choose **Run workflow**. Enter the measured distances and marks, select a fitting mode, adjust any other inputs as needed, then start the run. The evaluated marks appear in the run summary; the generated artifact contains the plot and run log.

See the [MIT License](LICENSE).
