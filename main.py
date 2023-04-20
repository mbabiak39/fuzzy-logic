import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define input variables
weight = ctrl.Antecedent(np.arange(0, 151, 1), 'weight')
age = ctrl.Antecedent(np.arange(0, 101, 1), 'age')

# Define output variable
dose = ctrl.Consequent(np.arange(0, 51, 1), 'dose')

# Define membership functions for weight
weight['low'] = fuzz.trapmf(weight.universe, [0, 0, 50, 70])
weight['medium'] = fuzz.trimf(weight.universe, [50, 80, 110])
weight['high'] = fuzz.trapmf(weight.universe, [90, 120, 150, 150])

# Define membership functions for age
age['young'] = fuzz.trimf(age.universe, [0, 0, 40])
age['middle'] = fuzz.trapmf(age.universe, [30, 50, 70, 70])
age['old'] = fuzz.trapmf(age.universe, [60, 80, 100, 100])

# Define membership functions for dose
dose['low'] = fuzz.trapmf(dose.universe, [0, 0, 10, 20])
dose['medium'] = fuzz.trimf(dose.universe, [10, 25, 40])
dose['high'] = fuzz.trapmf(dose.universe, [30, 40, 50, 50])

# Define rules for the dosage
rule1 = ctrl.Rule(weight['low'] & age['young'], dose['low'])
rule2 = ctrl.Rule(weight['low'] & age['middle'], dose['medium'])
rule3 = ctrl.Rule(weight['low'] & age['old'], dose['high'])
rule4 = ctrl.Rule(weight['medium'] & age['young'], dose['medium'])
rule5 = ctrl.Rule(weight['medium'] & age['middle'], dose['medium'])
rule6 = ctrl.Rule(weight['medium'] & age['old'], dose['high'])
rule7 = ctrl.Rule(weight['high'] & age['young'], dose['medium'])
rule8 = ctrl.Rule(weight['high'] & age['middle'], dose['high'])
rule9 = ctrl.Rule(weight['high'] & age['old'], dose['high'])

# Define the control system and add rules
medicine_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# Define a simulation to test the system
medicine_sim = ctrl.ControlSystemSimulation(medicine_ctrl)

# Set the patient weight and age
patient_weight = 75
patient_age = 50

# Pass the patient weight and age to the simulation
medicine_sim.input['weight'] = patient_weight
medicine_sim.input['age'] = patient_age

# Compute the recommended dosage
medicine_sim.compute()

# Print the output
print('Recommended dose:', medicine_sim.output['dose'])
