# SQuAD2.0 The Stanford Question Answering Dataset

# Setup
1. Make sure you have Miniconda installed
   1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
   2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
       
2. cd into src, run <code> conda env create -f environment.yml</code>
   1. This creates a Conda environment called <code>squad</code>

3. Run <code> conda activate squad </code>
   1. This activates the <code> squad </code> environment
   2. Do this each time you want to write/test your code

4. Run <code> python setup.py </code>
   1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
   2. This also pre-processes the dataset for efficient data loading
   3. For a MacBook Pro on the Stanford network, <code> setup.py </code> takes around 30 minutes total

5. Browse the code in <code> train.py </code>
   1. The <code> train.py </code> script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
   2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the <code> parser.add_argument </code> lines in the source code, or run <code> python train.py -n</code>.
