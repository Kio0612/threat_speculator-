**Threat Speculator**
 
**Mapping the Genome to Catch Dangerous Mutations and Speed Up Diagnoses**

**The Human Problem: Lab Technician Overload**
When doctors sequence a patient's DNA, they get back a massive list of thousands of mutations. The problem? Someone has to figure out which ones actually cause disease. Right now, lab technicians are forced to research these mutations one by one. It’s an exhausting, manual bottleneck that simply doesn't scale. 

Because of this overload, nearly half of all discovered mutations are dumped into a category called "Variants of Uncertain Significance" (VUS). Patients are left in the dark, and critical care is delayed. Older software tries to help, but it evaluates every mutation in complete isolation, ignoring how genetic changes actually interact in real life.

**Our Solution: Treating the Genome Like a Neighborhood**
Threat Speculator takes the guesswork out of the lab by modeling the human genome as a connected physical map. 
* Connecting the Dots: If mutations are located within 2,000 base pairs of each other, we link them together.
* Context is Everything: Using a GraphSAGE AI architecture, our model doesn't just judge a mutation by itself. It looks at the biological "danger" of its neighbors to make a smarter decision.
* Instant Answers: It automatically flags whether a genetic change is a pathogenic threat or just a harmless, natural variation.

**What's Under the Hood?**
* Data Crunching: Polars handles the massive genomic datasets at lightning speed.
* Feature Engineering: Scikit-Learn (FeatureHasher) translates biological traits into 78-dimensional math.
* The AI Brain: PyTorch and PyTorch Geometric power the graph neural network.

**Built for the Real World (Benchmarks)**
We engineered this pipeline to handle massive clinical datasets efficiently, utilizing PyTorch Geometric to evaluate genome-scale graphs on standard hardware.
- **Massive Scale:** Mapped out and evaluated 1.72 million mutations and nearly 30 million genomic connections.
- **Calibrated for Clinical Safety:** Because missing a disease-causing mutation is unacceptable, we mathematically optimized our decision threshold to 0.21. At this operating point, our model achieves **61.15% Recall** (catching over 60% of all true pathogenic threats) with a baseline ROC-AUC of 0.6207.
- **Workflow Acceleration:** Instead of blindly researching thousands of Variants of Uncertain Significance (VUS), lab technicians can use our model to instantly narrow the haystack. With a precision of 25.37%, roughly 1 in every 4 mutations the model flags is a confirmed threat, massively speeding up the manual diagnostic bottleneck.
