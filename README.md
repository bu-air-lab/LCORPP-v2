# Libraries required for the pipeline:
Pytorch: pip install torch torchvision
Matplotlib: python -m pip install -U matplotlib
Scipy:python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

# Run the abstract_simulator
python abstract_simulator.py
The following data documents wiil be generated:
reward_comps.csv,cost_comps.csv,successrate_comps.csv,peril_costs.csv,peril_rewards.csv,peril_successrates.csv,plearn_rewards.csv, plearn_costs.csv,rlearn_rewards.csv,rlearn_rewards.csv,rlearn_costs.csv
The following plots will be generated:
cost_comparison.pdf,cost_curve.pdf,cost_curveband.pdf, reward_comparison.pdf,reward_curve.pdf,reward_curveband.pdf,

# Module files:
dtio.py : functions for data output and plot
reasoner.py: functions of the reasoner 
interaction.py: functions of the POMDP planner
perception.py: functions of the classifier

# LCORPP-v2
To test the POMDP run the folwoing command:
python pomdp_tester.py>results.txt

The results.txt will contain the actions and results of each instance.
Three plots of "pomdp_r.pdf", "pomdp_c.pdf" and "pomdp_s.pdf" will be generated.
