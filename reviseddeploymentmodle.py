#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:49:43 2024

@author: matthewatwood

DAC Deployment Model with Monte Carlo Analysis
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy import stats
from matplotlib import gridspec
import json
import os
import shutil
from datetime import datetime
from collections import defaultdict

PARAMETER_INFO = {
    'start_year': {'description': 'Starting year for the deployment model',
                  'typical_range': '2024 to 2030',
                  'guidance': 'typically start from current/near-term year',
                  'min_value': 2024, 'max_value': 2099, 'default': 2024,
                  'monte_carlo': False,
                  'type': 'int'},  # Explicitly specify type
    'end_year': {'description': 'End year for the deployment model',
                'typical_range': '2040 to 2100',
                'guidance': 'Consider climate targets (2050, 2100)',
                'min_value': 2025, 'max_value': 2100, 'default': 2075,
                'monte_carlo': False,
                'type': 'int'},
    'midpoint_year': {'description': 'Year when growth rate is highest',
                     'typical_range': '2030 to 2040',
                     'guidance': 'Typically between start and end years',
                     'min_value': 2024, 'max_value': 2100, 'default': 2035,
                     'monte_carlo': True, 'monte_carlo_range': (2035, 2055),
                     'type': 'int'},
    'max_gt_capacity': {'description': 'Maximum annual CO2 removal capacity',
                       'typical_range': '1 to 20 GtCO2/year',
                       'guidance': 'Consider IPCC scenarios',
                       'min_value': 0.1, 'max_value': 100.0, 'default': 10.0,
                       'monte_carlo': True, 'monte_carlo_range': (20.0, 40.0),
                       'type': 'float'},
    'P0': {'description': 'Initial number of DAC units deployed',
           'typical_range': '1 to 100 units',
           'guidance': 'Based on current deployment plans',
           'min_value': 0.1, 'max_value': None, 'default': 2.0,
           'monte_carlo': True, 'monte_carlo_range': (1.0, 10.0),
           'type': 'float'},
    'r': {'description': 'Growth rate parameter',
          'typical_range': '0.2 to 1.0',
          'guidance': 'Higher values = faster adoption',
          'min_value': 0.01, 'max_value': 2.0, 'default': 0.5,
          'monte_carlo': True, 'monte_carlo_range': (0.2, 0.8),
          'type': 'float'},
    'v': {'description': 'Asymmetry parameter',
          'typical_range': '0.1 to 2.0',
          'guidance': '1.0 = symmetric growth',
          'min_value': 0.01, 'max_value': 5.0, 'default': 1.0,
          'monte_carlo': True, 'monte_carlo_range': (0.5, 1.5),
          'type': 'float'},
    'Q': {'description': 'Offset parameter',
          'typical_range': '0.1 to 2.0',
          'guidance': 'Shifts curve horizontally',
          'min_value': 0.01, 'max_value': 5.0, 'default': 1.0,
          'monte_carlo': True, 'monte_carlo_range': (0.5, 1.5),
          'type': 'float'},
    'tonnes_per_unit': {'description': 'CO2 removal per unit',
                       'typical_range': '500 to 5000 tonnes/year',
                       'guidance': 'Current designs ~1000 tonnes/year',
                       'min_value': 1.0, 'max_value': 10000.0, 'default': 1000.0,
                       'monte_carlo': True, 'monte_carlo_range': (800.0, 1200.0),
                       'type': 'float'},
    'service_life': {'description': 'Operating lifetime of each unit',
                    'typical_range': '10 to 30 years',
                    'guidance': 'Based on engineering design life',
                    'min_value': 1.0, 'max_value': 50.0, 'default': 20.0,
                    'monte_carlo': True, 'monte_carlo_range': (15.0, 25.0),
                    'type': 'float'},
    'capacity_factor': {'description': 'Operational capacity factor',
                       'typical_range': '0.8 to 0.95',
                       'guidance': 'Accounts for downtime',
                       'min_value': 0.5, 'max_value': 1.0, 'default': 0.9,
                       'monte_carlo': True, 'monte_carlo_range': (0.8, 0.95),
                       'type': 'float'},
    'learning_rate': {'description': 'Learning rate for efficiency',
                     'typical_range': '0.1 to 0.3',
                     'guidance': 'Improvement over time',
                     'min_value': 0.0, 'max_value': 0.5, 'default': 0.2,
                     'monte_carlo': True, 'monte_carlo_range': (0.1, 0.3),
                     'type': 'float'},
    'bg_growth_rate': {'description': 'Background emissions growth',
                      'typical_range': '0.5% to 2%',
                      'guidance': 'Historical rate ~1%',
                      'min_value': -0.05, 'max_value': 0.05, 'default': 0.01,
                      'monte_carlo': True, 'monte_carlo_range': (0.005, 0.015),
                      'type': 'float'},
    'decline_start_year': {'description': 'Emissions decline start',
                          'typical_range': '2030 to 2050',
                          'guidance': 'Based on policy scenarios',
                          'min_value': 2024, 'max_value': 2060, 'default': 2040,
                          'monte_carlo': True, 'monte_carlo_range': (2030, 2050),
                          'type': 'int'},
    'decline_rate': {'description': 'Annual emissions decline rate',
                    'typical_range': '1% to 4%',
                    'guidance': 'Based on policy scenarios',
                    'min_value': 0.0, 'max_value': 0.1, 'default': 0.02,
                    'monte_carlo': True, 'monte_carlo_range': (0.01, 0.04),
                    'type': 'float'},
    'capacity_improvement_rate': {
        'description': 'Annual improvement rate in capacity factor',
        'typical_range': '0 to 0.05 (0-5% per year)',
        'guidance': 'Rate at which capacity factor can improve through learning',
        'min_value': 0.0,
        'max_value': 0.10,
        'default': 0.01,
        'monte_carlo': True,
        'monte_carlo_range': (0.0, 0.05),
        'type': 'float'},
    'max_capacity_factor': {
        'description': 'Maximum achievable capacity factor',
        'typical_range': '1.0 to 1.5',
        'guidance': 'Ultimate limit for capacity factor improvement',
        'min_value': 0.8,
        'max_value': 2.0,
        'default': 1.2,
        'monte_carlo': True,
        'monte_carlo_range': (1.0, 1.5),
        'type': 'float'}
}

class ScenarioManager:
    def __init__(self, scenarios_file='dac_scenarios.json'):
        self.scenarios_file = os.path.abspath(scenarios_file)
        self.scenarios = self.load_scenarios()

    # Rest of the class implementation remains the same as before
    def load_scenarios(self):
        """Load scenarios with error handling."""
        try:
            if os.path.exists(self.scenarios_file):
                with open(self.scenarios_file, 'r') as f:
                    scenarios = json.load(f)
                if not isinstance(scenarios, dict):
                    raise ValueError("Invalid scenario file format")
                return scenarios
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Corrupt scenario file {self.scenarios_file}")
            return {}
        except Exception as e:
            print(f"Warning: Could not load scenarios: {str(e)}")
            return {}

    def save_scenario(self, name, params):
        """Save scenario with error handling and backup."""
        if not name or not isinstance(name, str):
            raise ValueError("Invalid scenario name")

        # Create backup of existing file
        if os.path.exists(self.scenarios_file):
            backup_file = f"{self.scenarios_file}.bak"
            try:
                shutil.copy2(self.scenarios_file, backup_file)
            except Exception as e:
                print(f"Warning: Could not create backup: {str(e)}")

        # Update scenarios and save
        try:
            params['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.scenarios[name] = params

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.scenarios_file), exist_ok=True)

            with open(self.scenarios_file, 'w') as f:
                json.dump(self.scenarios, f, indent=4)
        except Exception as e:
            print(f"Error saving scenario: {str(e)}")
            # Restore from backup if available
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, self.scenarios_file)
            raise

    def load_scenario(self, name):
        """Load a specific scenario with validation."""
        scenario = self.scenarios.get(name)
        if not scenario:
            raise KeyError(f"Scenario '{name}' not found")
        return scenario

    def list_scenarios(self):
        """List available scenarios with timestamps."""
        return [(name, data.get('timestamp', 'Unknown'))
                for name, data in self.scenarios.items()]

def print_results(results, model):
    """Print deployment model results."""
    print("\nDeployment Results:")
    print(f"Gigaton milestone year: {results['gigaton_milestone_year']}")
    print(f"Final yearly capacity: {results['yearly_capacity'][-1]/1e9:.2f} Gt/year")
    print(f"Total CO2 removed: {results['cumulative_removal'][-1]/1e9:.2f} Gt")
    print(f"Total units built: {int(results['cumulative_units_built'][-1])}")
    print(f"Final learning rate: {results['learning_rates'][-1]:.3f}")

def run_multiple_scenarios(n_scenarios=None):
    """Run and compare multiple scenarios with option for saved or new scenarios."""
    scenarios = []
    scenario_manager = ScenarioManager()  # Create ScenarioManager instance here

    if n_scenarios is None:
        n_scenarios = int(input("\nHow many scenarios would you like to compare? "))

    
    choice = st.radio("Scenario Comparison Options", options = ["Compare Saved Scenarios", "Create and Compare New Scenarios", "Mix of Saved and New Scenarios"])

    
    if choice == 'Compare Saved Scenarios':
        scenarios = load_saved_scenarios(scenario_manager, n_scenarios)
        st.write(scenarios)

    elif choice == 'Create and Compare New Scenarios':
        scenarios = create_new_scenarios(scenario_manager, n_scenarios)
    else:  # choice == 'Mix of Saved and New Scenarios'
        scenarios = mix_scenarios(scenario_manager, n_scenarios)
    
    return scenarios

def get_color_list(n):
    """Generate a list of distinct colors."""
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if n <= len(base_colors):
        return base_colors[:n]
    else:
        return [base_colors[i % len(base_colors)] for i in range(n)]

def get_plot_style():
    """Return common plot style settings."""
    return {
        'grid_props': {'linestyle': '--', 'alpha': 0.7},
        'title_props': {'fontsize': 12, 'pad': 10},
        'label_props': {'fontsize': 10},
        'tick_props': {'labelsize': 9},
        'legend_props': {'bbox_to_anchor': (1.05, 1), 'loc': 'upper left', 'fontsize': 9},
        'annotation_bbox': {'boxstyle': 'round,pad=0.3', 'fc': 'white', 'alpha': 0.9, 'linewidth': 0.5},
        'annotation_arrow': {'arrowstyle': '-|>', 'alpha': 0.6, 'linewidth': 1}
    }

class DACDeploymentModel:
    def __init__(self, **kwargs):
        """Initialize DACDeploymentModel with proper parameter handling."""
        # Initialize all parameters with defaults first
        self.start_year = 2024
        self.end_year = 2075
        self.midpoint_year = 2035
        self.decline_start_year = 2040
        self.max_gt_capacity = 10.0
        self.P0 = 2.0
        self.r = 0.5
        self.v = 1.0
        self.Q = 1.0
        self.tonnes_per_unit = 1000.0
        self.service_life = 20.0
        self.capacity_factor = 0.9
        self.learning_rate = 0.2
        self.bg_growth_rate = 0.01
        self.decline_rate = 0.02
        self.capacity_improvement_rate = 0.01
        self.max_capacity_factor = 1.2
    
        # Override defaults with any provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unexpected parameter {key}")
    
        # Validate parameters before proceeding
        if not self.validate_parameters(self.__dict__):
            raise ValueError("Parameter validation failed")
    
        # Calculate derived parameters
        self.max_capacity = self.max_gt_capacity * 1e9  # Convert Gt to tonnes
        self.K = int((self.max_capacity) / (self.tonnes_per_unit * self.capacity_factor))
        self.years = np.arange(self.start_year, self.end_year + 1)
        
        # Initialize capacity factor evolution
        self.capacity_factors = self.calculate_capacity_factor_evolution()

    def calculate_capacity_factor_evolution(self):
        """Calculate capacity factor improvement over time."""
        years_count = len(self.years)
        capacity_factors = np.zeros(years_count)
        
        # Start with initial capacity factor
        capacity_factors[0] = self.capacity_factor
        
        # Apply learning-based improvements with diminishing returns
        for i in range(1, years_count):
            potential_improvement = self.capacity_improvement_rate * (
                self.max_capacity_factor - capacity_factors[i-1]
            )
            actual_improvement = potential_improvement * np.exp(-i/10)  # Diminishing returns
            capacity_factors[i] = min(
                capacity_factors[i-1] + actual_improvement,
                self.max_capacity_factor
            )
        
        return capacity_factors

    def generalized_logistic(self, t):
        """Calculate the generalized logistic function value."""
        numerator = self.K
        denominator = (1 + self.Q * np.exp(-self.r * (t - self.midpoint_year))) ** (1/self.v)
        return numerator / denominator

    def calculate_learning_effect(self, cumulative_units):
        """Calculate learning effect with safety checks."""
        if cumulative_units < 0:
            raise ValueError("Cumulative units cannot be negative")
        
        if cumulative_units <= self.P0:
            return 1.0
        
        # Avoid division by zero or negative numbers
        safe_p0 = max(self.P0, 1e-10)
        return max((cumulative_units / safe_p0) ** (-self.learning_rate), 0.1)

    def calculate_deployment(self):
        """Calculate deployment trajectory and related metrics."""
        years_count = len(self.years)
        active_units = np.zeros(years_count, dtype=float)
        new_units = np.zeros(years_count, dtype=float)
        retired_units = np.zeros(years_count, dtype=float)
        yearly_capacity = np.zeros(years_count, dtype=float)
        cumulative_units_built = np.zeros(years_count, dtype=float)
    
        new_units[0] = self.P0
        active_units[0] = self.P0
        cumulative_units_built[0] = self.P0
    
        target_units = self.generalized_logistic(self.years)
    
        for year_idx in range(1, years_count):
            retirement_year = year_idx - int(self.service_life)
        
            if retirement_year >= 0 and retirement_year < year_idx:
                retired_units[year_idx] = float(new_units[retirement_year])
        
            current_active = active_units[year_idx - 1] - retired_units[year_idx]
            required_new = max(0, target_units[year_idx] - current_active)
        
            learning_rate = self.calculate_learning_effect(cumulative_units_built[year_idx-1])
            new_units[year_idx] = required_new * learning_rate
        
            cumulative_units_built[year_idx] = cumulative_units_built[year_idx-1] + new_units[year_idx]
            active_units[year_idx] = current_active + new_units[year_idx]
    
        yearly_capacity = active_units * self.tonnes_per_unit * self.capacity_factors
        cumulative_removal = np.cumsum(yearly_capacity)
    
        milestone_year = next((self.years[i] for i, cap in enumerate(yearly_capacity)
                             if cap >= 1e9), None)
        
        return {
            'years': self.years,
            'new_units': new_units,
            'active_units': active_units,
            'retired_units': retired_units,
            'yearly_capacity': yearly_capacity,
            'cumulative_removal': cumulative_removal,
            'gigaton_milestone_year': milestone_year,
            'cumulative_units_built': cumulative_units_built,
            'learning_rates': np.array([self.calculate_learning_effect(cu) for cu in cumulative_units_built]),
            'capacity_factors': self.capacity_factors
        }
    def calculate_carbon_budget(self, results):
        """Calculate carbon budget and warming probabilities with improved accuracy."""
        years_full = np.arange(2020, self.end_year + 1)
        baseline_emissions = np.zeros(len(years_full))
        dac_emissions = np.zeros(len(years_full))
        
        # Initialize baseline emissions with better precision
        baseline_emissions[0] = 36.4  # Starting emissions in 2020
        
        # Calculate baseline emissions with improved handling
        for i in range(1, len(years_full)):
            current_year = years_full[i]
            if current_year < self.decline_start_year:
                baseline_emissions[i] = baseline_emissions[i-1] * (1 + self.bg_growth_rate)
            else:
                baseline_emissions[i] = baseline_emissions[i-1] * (1 - self.decline_rate)
        
        # Improved DAC emissions calculation with proper alignment
        dac_start_idx = self.start_year - 2020
        if dac_start_idx < 0:
            raise ValueError("Start year cannot be before 2020")
        
        yearly_removals = results['yearly_capacity'] / 1e9
        end_idx = min(dac_start_idx + len(yearly_removals), len(dac_emissions))
        dac_emissions[dac_start_idx:end_idx] = -yearly_removals[:end_idx-dac_start_idx]
        
        # Calculate net emissions and cumulative values with better precision
        net_emissions = baseline_emissions + dac_emissions
        cumulative_emissions = np.cumsum(net_emissions)
        
        # Improved max atmospheric addition calculation
        max_atmospheric_addition = float(np.max(cumulative_emissions))
        
        # Calculate warming probabilities with bounds checking
        prob_1_5 = max(0, min(1, 1.9007 * np.exp(-0.00268 * max_atmospheric_addition)))
        prob_1_7 = max(0, min(1, 2.2763 * np.exp(-0.0018 * max_atmospheric_addition)))
        prob_2_0 = max(0, min(1, 2.405 * np.exp(-0.00116 * max_atmospheric_addition)))
        
        return {
            'years': years_full,
            'baseline_emissions': baseline_emissions,
            'dac_emissions': dac_emissions,
            'net_emissions': net_emissions,
            'cumulative_emissions': cumulative_emissions,
            'max_atmospheric_addition': max_atmospheric_addition,
            'prob_1_5C': prob_1_5,
            'prob_1_7C': prob_1_7,
            'prob_2_0C': prob_2_0
        }

    def plot_results(self):
        """Generate visualization of model results."""
        try:
            results = self.calculate_deployment()
            carbon_results = self.calculate_carbon_budget(results)
            
            fig = plt.figure(figsize=(15, 21))
            gs = gridspec.GridSpec(4, 2, figure=fig)
            
            self.plot_units_status(results, fig.add_subplot(gs[0, 0]))
            self.plot_new_units(results, fig.add_subplot(gs[0, 1]))
            self.plot_capacity(results, fig.add_subplot(gs[1, 0]))
            self.plot_cumulative_removal(results, fig.add_subplot(gs[1, 1]))
            self.plot_emissions_trajectory(carbon_results, fig.add_subplot(gs[2, 0]))
            self.plot_warming_probabilities(carbon_results, fig.add_subplot(gs[2, 1]))
            self.plot_capacity_factor_evolution(fig.add_subplot(gs[3, 0]))
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error generating plots: {str(e)}")
            return None

    def plot_units_status(self, results, ax):
        """Plot active and retired units over time."""
        ax.plot(self.years, results['active_units']/1e6, 'b-', label='Active Units')
        ax.plot(self.years, results['retired_units']/1e6, 'r--', label='Retired Units')
        ax.set_title('DAC Units Status')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Units (millions)')
        ax.grid(True)
        ax.legend()

    def plot_new_units(self, results, ax):
        """Plot new units added each year."""
        ax.bar(self.years, results['new_units']/1e6, color='g', alpha=0.6, label='New Units')
        ax.set_title('New DAC Units per Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Units Added (millions)')
        ax.grid(True)
        ax.legend()

    def plot_capacity(self, results, ax):
        """Plot yearly CO2 removal capacity."""
        ax.plot(self.years, results['yearly_capacity']/1e9, 'g-')
        ax.set_title('Annual CO2 Removal Capacity')
        ax.set_xlabel('Year')
        ax.set_ylabel('Capacity (Gt CO2/year)')
        ax.grid(True)

    def plot_cumulative_removal(self, results, ax):
        """Plot cumulative CO2 removal."""
        ax.plot(self.years, results['cumulative_removal']/1e9, 'b-')
        ax.set_title('Cumulative CO2 Removal')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Removal (Gt CO2)')
        ax.grid(True)

    def plot_emissions_trajectory(self, carbon_results, ax):
        """Plot emissions trajectories."""
        years = carbon_results['years']
        ax.plot(years, carbon_results['baseline_emissions'], 'r-', label='Baseline')
        ax.plot(years, carbon_results['net_emissions'], 'g-', label='With DAC')
        ax.plot(years, carbon_results['dac_emissions'], 'b--', label='DAC Contribution')
        ax.set_title('Emissions Trajectory')
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual Emissions (Gt CO2/year)')
        ax.grid(True)
        ax.legend()

    def plot_warming_probabilities(self, carbon_results, ax):
        """Plot warming scenario probabilities."""
        labels = ['1.5°C', '1.7°C', '2.0°C']
        probs = [carbon_results['prob_1_5C'], 
                carbon_results['prob_1_7C'],
                carbon_results['prob_2_0C']]
        
        ax.bar(labels, probs, color=['green', 'yellow', 'red'], alpha=0.6)
        ax.set_title('Probability of Staying Below Temperature Thresholds')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)

    def plot_capacity_factor_evolution(self, ax):
        """Plot capacity factor evolution over time."""
        ax.plot(self.years, self.capacity_factors, 'g-')
        ax.set_title('Capacity Factor Evolution')
        ax.set_xlabel('Year')
        ax.set_ylabel('Capacity Factor')
        ax.grid(True)

    @staticmethod
    def validate_parameters(params):
        """Validate parameter combinations and relationships with adjusted criteria."""
        try:
            # Timeline checks
            if params['midpoint_year'] <= params['start_year']:
                print(f"Failed timeline check: midpoint_year ({params['midpoint_year']}) <= start_year ({params['start_year']})")
                return False
            if params['end_year'] <= params['midpoint_year']:
                print(f"Failed timeline check: end_year ({params['end_year']}) <= midpoint_year ({params['midpoint_year']})")
                return False
                
            # More lenient growth rate validation
            years_to_mid = params['midpoint_year'] - params['start_year']
            implied_annual_growth = (params['r'] / years_to_mid) * 100  # Convert to percentage
            
            max_allowed_growth = 10.0 if years_to_mid < 10 else (
                8.0 if years_to_mid < 15 else (
                6.0 if years_to_mid < 20 else 5.0))
            
            if implied_annual_growth > max_allowed_growth:
                print(f"Failed growth check: implied annual growth {implied_annual_growth:.1f}% > {max_allowed_growth:.1f}% maximum")
                return False
                
            # Capacity checks
            if params.get('capacity_factor', 0) <= 0:
                print("Failed capacity check: capacity_factor <= 0")
                return False
            if params.get('capacity_factor', 0) > params.get('max_capacity_factor', 1.2):
                print(f"Failed capacity check: capacity_factor ({params['capacity_factor']}) > max_capacity_factor ({params['max_capacity_factor']})")
                return False
                
            # Unit checks
            if params.get('tonnes_per_unit', 0) <= 0:
                print("Failed unit check: tonnes_per_unit <= 0")
                return False
            
            # Initial deployment check
            initial_capacity = params['P0'] * params['tonnes_per_unit'] * params['capacity_factor']
            max_capacity = params['max_gt_capacity'] * 1e9  # Convert to tonnes
            if initial_capacity > max_capacity * 0.1:  # Initial deployment shouldn't exceed 10% of max
                print(f"Failed initial deployment check: {initial_capacity/1e6:.1f} Mt > {(max_capacity * 0.1)/1e6:.1f} Mt maximum")
                return False
                
            return True
            
        except Exception as e:
            print(f"Parameter validation error: {str(e)}")
            return False

def run_monte_carlo(n_simulations=1000):
    results_df = pd.DataFrame()
    parameter_list = []
    
    for i in range(n_simulations):
        try:
            params = generate_random_parameters()
            model = DACDeploymentModel(**params)
            results_i = model.calculate_deployment()
            carbon_results = model.calculate_carbon_budget(results_i)
            
            # Store results with correct column names
            row_data = {
                'Max Atmospheric Addition (Gt)': carbon_results['max_atmospheric_addition'],
                'Total Removal by 2075 (Gt)': results_i['cumulative_removal'][-1]/1e9,
                'Milestone Year': results_i['gigaton_milestone_year'],
                'Max Deployment Rate': np.max(results_i['new_units']),
                'Final Capacity': results_i['yearly_capacity'][-1]/1e9,
                'prob_1_5C': carbon_results['prob_1_5C'],
                'prob_1_7C': carbon_results['prob_1_7C'],
                'prob_2_0C': carbon_results['prob_2_0C']
            }
            
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
            parameter_list.append(params)
            
            if i % 100 == 0:
                st.write(f"Completed {i} simulations")
        except Exception as e:
            st.error(f"Error in simulation {i}: {str(e)}")
            continue
    
    for param in PARAMETER_INFO:
        if PARAMETER_INFO[param].get('monte_carlo', False):
            results_df[f'Param_{param}'] = [p[param] for p in parameter_list]
    
    summary_stats = calculate_summary_statistics(results_df)
    correlation_matrix = results_df.corr()
    
    return results_df, summary_stats, correlation_matrix
    
    print("\nKey Parameter Correlations:")
    param_cols = [col for col in correlation_matrix.index if col.startswith('Param_')]
    metric_cols = ['Max Atmospheric Addition (Gt)', 'Total Removal by 2075 (Gt)']
    
    try:
        for param in param_cols:
            for metric in metric_cols:
                corr = correlation_matrix.loc[param, metric]
                print(f"{param} vs {metric}: {corr:.3f}")
    except Exception as e:
        print(f"Error calculating correlations: {str(e)}")

def plot_correlation_heatmap(results_df):
    """Plot correlation heatmap with proper column handling."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    param_cols = [col for col in results_df.columns if col.startswith('Param_')]
    metric_cols = ['Max Atmospheric Addition (Gt)', 'Total Removal by 2075 (Gt)']
    
    # Ensure all columns exist
    available_metrics = [col for col in metric_cols if col in results_df.columns]
    if not available_metrics:
        print("Warning: No matching metric columns found for correlation heatmap")
        return None
    
    corr_data = results_df[param_cols + available_metrics].corr()
    relevant_corr = corr_data.loc[param_cols, available_metrics]
    
    sns.heatmap(relevant_corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Parameter-Metric Correlations')
    plt.tight_layout()
    
    return fig

def generate_random_parameters():
    """Generate random parameters within defined ranges."""
    params = {}
    for param, info in PARAMETER_INFO.items():
        if info.get('monte_carlo', False):
            if 'monte_carlo_range' in info:
                min_val, max_val = info['monte_carlo_range']
                if isinstance(info['default'], int):
                    params[param] = int(np.random.uniform(min_val, max_val))
                else:
                    params[param] = np.random.uniform(min_val, max_val)
            else:
                params[param] = info['default']
        else:
            params[param] = info['default']
    return params

def generate_random_parameters_with_bounds(bounds):
    """Generate random parameters with proper handling of fixed parameters."""
    params = {}
    
    # First set fixed parameters that shouldn't be randomized
    params['start_year'] = PARAMETER_INFO['start_year']['default']
    params['end_year'] = PARAMETER_INFO['end_year']['default']
    
    # Then set timeline parameters that depend on fixed parameters
    if 'midpoint_year' in bounds:
        min_val, max_val = bounds['midpoint_year']
        # Ensure midpoint is between start and end years
        adjusted_min = max(min_val, params['start_year'] + 5)
        adjusted_max = min(max_val, params['end_year'] - 5)
        params['midpoint_year'] = int(np.random.uniform(adjusted_min, adjusted_max + 1))
    else:
        params['midpoint_year'] = PARAMETER_INFO['midpoint_year']['default']
        
    if 'decline_start_year' in bounds:
        min_val, max_val = bounds['decline_start_year']
        # Ensure decline start is after start year
        adjusted_min = max(min_val, params['start_year'])
        adjusted_max = min(max_val, params['end_year'])
        params['decline_start_year'] = int(np.random.uniform(adjusted_min, adjusted_max + 1))
    else:
        params['decline_start_year'] = PARAMETER_INFO['decline_start_year']['default']
    
    # Set remaining parameters
    for param, info in PARAMETER_INFO.items():
        if param not in params and param in bounds:
            min_val, max_val = bounds[param]
            if info.get('type') == 'int':
                params[param] = int(np.random.uniform(min_val, max_val + 1))
            else:
                params[param] = np.random.uniform(min_val, max_val)
        elif param not in params:
            params[param] = info['default']
    
    return params

def store_monte_carlo_results(results, results_i, carbon_results, params, model):
    results['parameters'].append(params)
    results['milestone_years'].append(
        results_i['gigaton_milestone_year'] if results_i['gigaton_milestone_year'] else np.nan
    )
    results['max_deployment_rates'].append(np.max(results_i['new_units']))
    results['final_capacities'].append(results_i['yearly_capacity'][-1] / 1e9)
    results['total_removals'].append(results_i['cumulative_removal'][-1] / 1e9)
    
    peak_year_idx = np.argmax(results_i['new_units'])
    results['peak_deployment_years'].append(model.years[peak_year_idx])
    results['time_to_peak'].append(model.years[peak_year_idx] - model.start_year)
    results['total_units_built'].append(results_i['cumulative_units_built'][-1])
    results['learning_effects'].append(results_i['learning_rates'][-1])
    results['replacement_rates'].append(np.max(results_i['retired_units']))
    
    results['max_atmospheric_addition'].append(carbon_results['max_atmospheric_addition'])
    results['prob_1_5C'].append(carbon_results['prob_1_5C'])
    results['prob_1_7C'].append(carbon_results['prob_1_7C'])
    results['prob_2_0C'].append(carbon_results['prob_2_0C'])

def process_monte_carlo_results(results):
    if not results['parameters']:
        raise ValueError("No successful simulations completed")
        
    df_data = create_results_dataframe(results)
    results_df = pd.DataFrame(df_data)
    
    summary_stats = calculate_summary_statistics(results_df)
    correlation_matrix = results_df.corr()
    
    return results_df, summary_stats, correlation_matrix

def run_cost_analysis(param_df):
    cost_params = {
        'capex_per_unit': 1000000,
        'opex_per_tonne': 200,
        'learning_rate': 0.15,
        'financing_rate': 0.07,
        'project_lifetime': 20
    }
    
    results = {
        'yearly_costs': [],
        'cumulative_investment': [],
        'levelized_cost': [],
        'cost_breakdown': [],
        'learning_curve_savings': []
    }
    
    for _, row in param_df.iterrows():
        cumulative_units = np.cumsum(row['new_units'])
        learning_factor = (cumulative_units / row['P0']) ** (-np.log2(1 - cost_params['learning_rate']))
        unit_capex = cost_params['capex_per_unit'] * learning_factor
        
        yearly_capex = row['new_units'] * unit_capex
        yearly_opex = row['yearly_capacity'] * cost_params['opex_per_tonne']
        total_yearly_cost = yearly_capex + yearly_opex
        
        discount_factors = (1 + cost_params['financing_rate']) ** -np.arange(len(total_yearly_cost))
        npv = np.sum(total_yearly_cost * discount_factors)
        levelized_cost = npv / np.sum(row['yearly_capacity'] * discount_factors)
        
        results['yearly_costs'].append(total_yearly_cost)
        results['cumulative_investment'].append(np.cumsum(total_yearly_cost))
        results['levelized_cost'].append(levelized_cost)
        results['cost_breakdown'].append({
            'capex': np.sum(yearly_capex),
            'opex': np.sum(yearly_opex)
        })
        results['learning_curve_savings'].append(
            np.sum(row['new_units'] * cost_params['capex_per_unit']) - np.sum(yearly_capex)
        )
    
    return results

def run_timeline_analysis(param_df):
    """Run timeline analysis with Monte Carlo results."""
    n_simulations = 100
    results = {
        'buildout_rates': [],
        'milestone_years': [],
        'manufacturing_capacity': [],
        'supply_chain_requirements': [],
        'critical_path_metrics': []
    }
    
    for i in range(n_simulations):
        try:
            params = generate_random_parameters()
            model = DACDeploymentModel(**params)
            results_i = model.calculate_deployment()
            
            yearly_buildout = np.diff(results_i['new_units'])
            manufacturing_params = {
                'ramp_up_time': 2,
                'factory_capacity': 100,
                'supply_chain_lag': 1
            }
            
            required_factories = np.ceil(yearly_buildout / manufacturing_params['factory_capacity'])
            material_requirements = calculate_material_requirements(yearly_buildout)
            
            store_timeline_results(results, yearly_buildout, required_factories,
                                 material_requirements, manufacturing_params)
            
            if i % 10 == 0:
                st.write(f"Completed {i} timeline simulations")
            
        except Exception as e:
            print(f"Error in timeline simulation {i}: {str(e)}")
            continue
    
    st.write(f"\nCompleted {n_simulations} timeline simulations")
    fig = plot_timeline_analysis(results)
    st.pyplot(fig)
    
    return results

def store_timeline_results(results, yearly_buildout, required_factories, 
                         material_requirements, manufacturing_params):
    """Store timeline analysis results."""
    results['buildout_rates'].append({
        'max_rate': np.max(yearly_buildout),
        'sustained_rate': np.mean(yearly_buildout),
        'volatility': np.std(yearly_buildout)
    })
    
    results['manufacturing_capacity'].append({
        'max_factories': np.max(required_factories),
        'factory_timeline': required_factories
    })
    
    results['supply_chain_requirements'].append(material_requirements)
    
    results['critical_path_metrics'].append({
        'factory_construction_time': np.sum(required_factories > 0) * manufacturing_params['ramp_up_time'],
        'supply_chain_development': np.max(yearly_buildout) * manufacturing_params['supply_chain_lag']
    })


def plot_removal_distribution(results_df, ax):
    """Plot distribution of total CO2 removal."""
    sns.histplot(data=results_df['Total Removal by 2075 (Gt)'], bins=30, ax=ax)
    median = results_df['Total Removal by 2075 (Gt)'].median()
    ax.axvline(median, color='r', linestyle='-', 
               label=f'Median: {median:.1f} Gt')
    ax.set_title('Distribution of Total CO2 Removal')
    ax.set_xlabel('Total Removal by 2075 (Gt)')
    ax.legend()

def plot_warming_probabilities_distribution(results_df, ax):
    """Plot distribution of warming probabilities with updated boxplot parameters."""
    prob_cols = ['prob_1_5C', 'prob_1_7C', 'prob_2_0C']
    labels = ['1.5°C', '1.7°C', '2.0°C']
    
    box_data = [results_df[col] for col in prob_cols]
    ax.boxplot(box_data, tick_labels=labels)  # Updated parameter name
    ax.set_title('Distribution of Temperature Threshold Probabilities')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)

def plot_sensitivity_analysis(results_df):
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
    axes = axes.flatten()
    
    sensitivity_params = ['r', 'service_life', 'capacity_factor', 
                         'learning_rate', 'bg_growth_rate', 'decline_rate']
    
    for i, param in enumerate(sensitivity_params):
        param_name = f'Param_{param}'
        if param_name in results_df.columns:
            sns.regplot(data=results_df, x=param_name, y='Max Atmospheric Addition (Gt)',
                       ax=axes[i], scatter_kws={'alpha':0.5}, ci=90)
            axes[i].set_title(f'{param} vs Maximum Atmospheric Addition')
            
            corr = results_df[param_name].corr(results_df['Max Atmospheric Addition (Gt)'])
            axes[i].text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                        transform=axes[i].transAxes, verticalalignment='top')
    
    plt.tight_layout()
    return fig

def plot_milestone_distribution(results_df, summary_stats, ax):
    sns.histplot(data=results_df['Milestone Year'].dropna(), bins=30, ax=ax)
    if 'Milestone Year' in summary_stats:
        ax.axvline(summary_stats['Milestone Year']['p50'], color='r', linestyle='-',
                  label=f"Median: {summary_stats['Milestone Year']['p50']:.1f}")

def plot_atmospheric_addition(results_df, ax):
    sns.histplot(data=results_df['Max Atmospheric Addition (Gt)'], bins=30, ax=ax)
    median = results_df['Max Atmospheric Addition (Gt)'].median()
    ax.axvline(median, color='r', linestyle='-', label=f'Median: {median:.1f}')

def plot_cost_analysis(cost_results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    plot_levelized_cost_distribution(cost_results, ax1)
    plot_learning_curve_savings(cost_results, ax2)
    plot_cost_breakdown(cost_results, ax3)
    plot_investment_timeline(cost_results, ax4)
    
    plt.tight_layout()
    return fig

def create_results_dataframe(results):
    df_data = {
        'Milestone Year': results['milestone_years'],
        'Max Deployment Rate (units/year)': results['max_deployment_rates'],
        'Final Capacity (Gt/year)': results['final_capacities'],
        'Total Removal by 2075 (Gt)': results['total_removals']
        # Add other metrics as needed
    }
    return df_data

def calculate_summary_statistics(results_df):
    summary_stats = {}
    for column in results_df.columns:
        data = results_df[column].dropna()
        if len(data) > 0:
            summary_stats[column] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'p10': np.percentile(data, 10),
                'p50': np.percentile(data, 50),
                'p90': np.percentile(data, 90)
            }
    return summary_stats

def validate_parameter_bounds(bounds):
    """Validate parameter bounds with more lenient thresholds."""
    validation_rules = {
        'midpoint_year': {
            'dependencies': ['start_year', 'end_year'],
            'rule': lambda params: (params['midpoint_year'] - params['start_year'] >= 5 and
                                  params['end_year'] - params['midpoint_year'] >= 5),
            'message': 'Midpoint year should be at least 5 years from start and end years'
        },
        'r': {
            'dependencies': ['midpoint_year', 'start_year'],
            'rule': lambda params: params['r'] * (params['midpoint_year'] - params['start_year']) <= 4.0,
            'message': 'Growth rate too aggressive for given timeline'
        }
    }
    
    validation_results = {
        'passed': True,
        'warnings': [],
        'parameter_status': {}
    }
    
    for param, rules in validation_rules.items():
        try:
            if param in bounds:
                min_val, max_val = bounds[param]
                test_params = {param: (min_val + max_val) / 2}
                
                for dep in rules['dependencies']:
                    if dep in bounds:
                        test_params[dep] = sum(bounds[dep]) / 2
                    else:
                        test_params[dep] = PARAMETER_INFO[dep]['default']
                
                if not rules['rule'](test_params):
                    validation_results['warnings'].append(
                        f"Warning for {param}: {rules['message']}")
                    validation_results['parameter_status'][param] = 'warning'
                else:
                    validation_results['parameter_status'][param] = 'ok'
                    
        except Exception as e:
            validation_results['warnings'].append(
                f"Error validating {param}: {str(e)}")
            validation_results['parameter_status'][param] = 'error'
    
    return validation_results

def generate_detailed_report(results_df, summary_stats, correlation_matrix, bounds, validation_results):
    """Generate detailed analysis report with enhanced statistics and insights."""
    report = {
        'overview': {
            'total_simulations': len(results_df),
            'failed_simulations': results_df.isna().any(axis=1).sum(),
            'completion_rate': (1 - results_df.isna().any(axis=1).sum() / len(results_df)) * 100
        },
        'parameter_analysis': {},
        'key_metrics': {},
        'correlations': {},
        'sensitivity': {},
        'validation': validation_results
    }
    
    # Parameter analysis
    for param in [col for col in results_df.columns if col.startswith('Param_')]:
        param_name = param.replace('Param_', '')
        param_data = results_df[param].dropna()
        
        report['parameter_analysis'][param_name] = {
            'mean': param_data.mean(),
            'std': param_data.std(),
            'min': param_data.min(),
            'max': param_data.max(),
            'specified_bounds': bounds.get(param_name, ('default', 'default')),
            'validation_status': validation_results['parameter_status'].get(param_name, 'not_validated')
        }
    
    # Key metrics analysis
    metric_cols = ['Max Atmospheric Addition (Gt)', 'Total Removal by 2075 (Gt)',
                  'prob_1_5C', 'prob_1_7C', 'prob_2_0C']
    
    for metric in metric_cols:
        if metric in results_df.columns:
            data = results_df[metric].dropna()
            report['key_metrics'][metric] = {
                'mean': data.mean(),
                'std': data.std(),
                'p10': data.quantile(0.1),
                'p50': data.quantile(0.5),
                'p90': data.quantile(0.9)
            }
    
    # Calculate success rates for temperature targets
    for prob_col in ['prob_1_5C', 'prob_1_7C', 'prob_2_0C']:
        if prob_col in results_df.columns:
            threshold = 0.5  # Example threshold
            success_rate = (results_df[prob_col] > threshold).mean()
            report['key_metrics'][f'{prob_col}_success_rate'] = success_rate
    
    return report

def print_detailed_report(report):
    """Print the detailed analysis report in a formatted way."""
    print("\n=== Monte Carlo Analysis Detailed Report ===\n")
    
    print("Overview:")
    print(f"Total simulations: {report['overview']['total_simulations']}")
    print(f"Completion rate: {report['overview']['completion_rate']:.1f}%")
    print(f"Failed simulations: {report['overview']['failed_simulations']}")
    
    print("\nParameter Analysis:")
    for param, stats in report['parameter_analysis'].items():
        print(f"\n{param}:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std Dev: {stats['std']:.3f}")
        print(f"  Range: {stats['min']:.3f} to {stats['max']:.3f}")
        print(f"  Specified bounds: {stats['specified_bounds']}")
        print(f"  Validation status: {stats['validation_status']}")
    
    print("\nKey Metrics:")
    for metric, stats in report['key_metrics'].items():
        if not metric.endswith('success_rate'):
            print(f"\n{metric}:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Std Dev: {stats['std']:.3f}")
            print(f"  P10-P90 Range: {stats['p10']:.3f} to {stats['p90']:.3f}")
    
    print("\nTemperature Target Success Rates:")
    for metric, value in report['key_metrics'].items():
        if metric.endswith('success_rate'):
            temp = metric.split('_')[1]
            print(f"{temp}: {value:.1%} probability of success")
    
    print("\nValidation Warnings:")
    for warning in report['validation']['warnings']:
        print(f"- {warning}")

def run_target_based_analysis():
    
    with st.form(key="Target_Monte_Carlo"):
        """Run target-based Monte Carlo analysis with enhanced validation and reporting."""
        
        def print_progress(iteration, total, prefix='Progress', suffix='Complete', decimals=1, length=50):
            """Print a progress bar for long-running operations."""
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filled_length = int(length * iteration // total)
            bar = '█' * filled_length + '-' * (length - filled_length)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
            if iteration == total:
                st.write()
        
        # Define temperature Choices
        temp_choice = st.radio("\nSelect temperature target (°C):", [1.5, 1.7, 2])

        # Get analysis parameters
        target_temp = temp_choice

        target_prob = st.number_input(
            f"Desired probability (0-1) of avoiding choice of temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

        n_sims_input = st.number_input(
            "Number of simulations:",
            min_value=100,
            max_value=100000,
            value=1000
        )
        n_sims = int(n_sims_input)

        # Get and validate parameter bounds
        bounds = get_parameter_bounds()
        validation_results = validate_parameter_bounds(bounds)

        # Submit button
        submit = st.form_submit_button("Generate Simulation")
        if submit:
            try:
                if validation_results['warnings']:
                    print("\nValidation Warnings:")
                    for warning in validation_results['warnings']:
                        st.warning(f"- {warning}")
                    # Display the proceed message
                    proceeding = st.radio("Proceed with analysis despite warnings? (Click Generate Simulation After Deciding)", ["Hold", "Yes", "No"])

                    # Add a button to allow the user to proceed
                    if proceeding == "Yes":
                        st.write("Proceeding with analysis...")
                         # Run Monte Carlo analysis with progress tracking
                        st.write(f"\nRunning {n_sims} Monte Carlo simulations...")
                        results = run_bounded_monte_carlo(n_sims, bounds, progress_callback=print_progress)
                        results_df, summary_stats, correlation_matrix = results
                        
                        # Print results
                        st.write(f"\nResults for {target_temp}°C target:")
                        if target_temp == 2:
                            prob_col = f"prob_{str(target_temp)}_0C"
                        else:
                            prob_col = f"prob_{str(target_temp).replace('.', '_')}C"
                        actual_prob = results_df[prob_col].mean()
                        st.write(f"Probability of staying below {target_temp}°C: {actual_prob:.3f}")
                        st.write(f"Target probability: {target_prob:.3f}")
                        st.write(f"Gap: {(target_prob - actual_prob):.3f}")
                        
                        # Print parameter ranges used
                        st.write("\nParameter Ranges Used:")
                        for param, (min_val, max_val) in bounds.items():
                            actual_min = results_df[f'Param_{param}'].min()
                            actual_max = results_df[f'Param_{param}'].max()
                            st.write(f"{param}:")
                            st.write(f"  Specified range: {min_val:.3f} to {max_val:.3f}")
                            st.write(f"  Actual range: {actual_min:.3f} to {actual_max:.3f}")
                        
                        # Generate plots
                        st.write("\nGenerating plots...")
                        try:
                            figs = plot_monte_carlo_results(results_df, summary_stats, correlation_matrix)
                            for i, fig in enumerate(figs, 1):
                                st.write(f"Displaying plot {i} of {len(figs)}...")
                                plt.figure(fig.number)
                                st.pyplot(fig)
                                plt.close(fig)
                        except Exception as e:
                            print(f"Warning: Error generating some plots: {str(e)}")
                        
                        return results_df, summary_stats, correlation_matrix
                        

                    elif proceeding == "No":
                        st.write("Analysis aborted.")
                        return None
                    else:
                        st.write("Please make a decision")
            except Exception as e:
                        st.error(f"Error in Monte-Carlo Analysis Exception: {str(e)}")
                
                
               

def get_parameter_bounds():
    """Get user input for parameter bounds."""
    bounds = {}
    print("\nEnter parameter bounds for Monte Carlo analysis.")
    print("Press Enter to use default bounds for any parameter.\n")
    
    # Group parameters by category for better organization
    parameter_groups = {
        'Timeline Parameters': ['midpoint_year', 'decline_start_year'],
        'Capacity Parameters': ['max_gt_capacity', 'tonnes_per_unit', 'capacity_factor'],
        'Growth Parameters': ['P0', 'r', 'v', 'Q'],
        'Learning Parameters': ['service_life', 'learning_rate'],
        'Emissions Parameters': ['bg_growth_rate', 'decline_rate']
    }

    # Page title
    st.title("Parameter Range Configuration")

    # Collecting user inputs
    inputs = {}

    for group_name, params in parameter_groups.items():
        st.header(group_name)
        for param in params:
            if PARAMETER_INFO[param].get('monte_carlo', False):
                info = PARAMETER_INFO[param]
                description = info.get('description', 'No description available')
                min_value, max_value = info.get('monte_carlo_range', (info['min_value'], info['max_value']))
                
                st.subheader(f"{param}")
                st.write(f"**Description:** {description}")

                # Input fields for custom minimum and maximum values
                col1, col2 = st.columns(2)
                
                with col1:
                    if type(min_value) == int:
                        inputs[f"{param}_min"], inputs[f"{param}_max"] = st.slider(
                            f"Minimum {param}:",
                            min_value = min_value,
                            max_value = max_value,
                            value = (min_value, max_value),
                            key=f"{param}_range"
                        )
                    else:
                        inputs[f"{param}_min"], inputs[f"{param}_max"] = st.slider(
                            f"Minimum {param}:",
                            min_value = min_value,
                            max_value = max_value,
                            value = (min_value, max_value),
                            step = .001,
                            key=f"{param}_range"
                        )

                bounds[param] = (inputs[f"{param}_min"], inputs[f"{param}_max"])
    # Display final inputs in a clear layout
    st.write("### Final Input Values")
    st.json(inputs)



    return bounds

def run_bounded_monte_carlo(n_simulations, bounds, progress_callback=None):
    """Run Monte Carlo analysis with enhanced error handling and debug output."""
    results_df = pd.DataFrame()
    parameter_list = []
    valid_simulations = 0
    failed_simulations = 0
    debug_info = []
    
    for i in range(n_simulations):
        try:
            # Generate parameters
            params = generate_random_parameters_with_bounds(bounds)
            
            # Debug output for parameters
            debug_info.append(f"\nSimulation {i} parameters:")
            for key, value in params.items():
                debug_info.append(f"{key}: {value}")
            
            # Parameter validation checks
            if not DACDeploymentModel.validate_parameters(params):  # Use class method
                debug_info.append("Parameter validation failed")
                failed_simulations += 1
                continue
            
            # Create model and run calculations
            try:
                model = DACDeploymentModel(**params)
                results_i = model.calculate_deployment()
                
                if results_i is None:
                    debug_info.append("calculate_deployment returned None")
                    failed_simulations += 1
                    continue
                    
                carbon_results = model.calculate_carbon_budget(results_i)
                
                if carbon_results is None:
                    debug_info.append("calculate_carbon_budget returned None")
                    failed_simulations += 1
                    continue
                
                # Extract results with validation
                try:
                    row_data = {
                        'Max Atmospheric Addition (Gt)': float(carbon_results['max_atmospheric_addition']),
                        'Total Removal by 2075 (Gt)': float(results_i['cumulative_removal'][-1]/1e9),
                        'Milestone Year': results_i['gigaton_milestone_year'],
                        'Max Deployment Rate': float(np.max(results_i['new_units'])),
                        'Final Capacity': float(results_i['yearly_capacity'][-1]/1e9),
                        'prob_1_5C': float(carbon_results['prob_1_5C']),
                        'prob_1_7C': float(carbon_results['prob_1_7C']),
                        'prob_2_0C': float(carbon_results['prob_2_0C'])
                    }
                    
                    # Validate results
                    if any(pd.isna(v) for v in row_data.values() if v is not None):
                        debug_info.append("Results contain NaN values")
                        failed_simulations += 1
                        continue
                        
                    results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
                    parameter_list.append(params)
                    valid_simulations += 1
                    
                except Exception as e:
                    debug_info.append(f"Error extracting results: {str(e)}")
                    failed_simulations += 1
                    continue
                    
            except Exception as e:
                debug_info.append(f"Error in model calculations: {str(e)}")
                failed_simulations += 1
                continue
            
            # Update progress
            if progress_callback and i % 10 == 0:
                progress_callback(i + 1, n_simulations)
                
        except Exception as e:
            debug_info.append(f"Error in simulation {i}: {str(e)}")
            failed_simulations += 1
            continue
    
    # Final progress update
    if progress_callback:
        progress_callback(n_simulations, n_simulations)
    
    # Print debug summary
    print("\nSimulation Debug Summary:")
    print(f"Total simulations attempted: {n_simulations}")
    print(f"Valid simulations: {valid_simulations}")
    print(f"Failed simulations: {failed_simulations}")
    print(f"Success rate: {(valid_simulations/n_simulations)*100:.1f}%")
    
    if failed_simulations > 0:
        print("\nDebug Information:")
        print("\n".join(debug_info[-10:]))  # Show last 10 debug messages
        
    if valid_simulations > 0:
        # Add parameter columns to results
        for param in PARAMETER_INFO:
            if PARAMETER_INFO[param].get('monte_carlo', False):
                results_df[f'Param_{param}'] = [p[param] for p in parameter_list]
        
        # Calculate summary statistics
        summary_stats = calculate_summary_statistics(results_df)
        correlation_matrix = results_df.corr()
        
        return results_df, summary_stats, correlation_matrix
    else:
        print("\nFull Debug Log:")
        print("\n".join(debug_info))
        raise ValueError("No valid simulations completed. Try adjusting parameter bounds.")

def plot_monte_carlo_results(results_df, summary_stats, correlation_matrix):
    """Plot comprehensive Monte Carlo analysis results with improved error handling."""
    figs = []
    
    try:
        # Distribution plots
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig1.suptitle('Key Metrics Distributions', fontsize=16)
        
        plot_milestone_distribution(results_df, summary_stats, ax1)
        plot_atmospheric_addition(results_df, ax2)
        plot_removal_distribution(results_df, ax3)
        plot_warming_probabilities_distribution(results_df, ax4)
        plt.tight_layout()
        figs.append(fig1)
        
        # Sensitivity analysis
        try:
            fig2 = plot_sensitivity_analysis(results_df)
            if fig2 is not None:
                figs.append(fig2)
        except Exception as e:
            print(f"Warning: Could not generate sensitivity analysis plot: {str(e)}")
        
        # Correlation heatmap
        try:
            fig3 = plot_correlation_heatmap(results_df)
            if fig3 is not None:
                figs.append(fig3)
        except Exception as e:
            print(f"Warning: Could not generate correlation heatmap: {str(e)}")
        
    except Exception as e:
        print(f"Warning: Error in plot generation: {str(e)}")
    
    return figs

def plot_monte_carlo_distributions(results_df, summary_stats):
    """Plot Monte Carlo distribution metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Key Metrics Distributions', fontsize=16)
    
    plot_milestone_distribution(results_df, summary_stats, ax1)
    plot_atmospheric_addition(results_df, ax2)
    plot_removal_distribution(results_df, ax3)
    plot_warming_probabilities_distribution(results_df, ax4)
    
    plt.tight_layout()
    return fig

def get_user_inputs(scenario_manager):
    """Get parameter inputs from user with validation."""
    params = {}
    print("\nEnter parameters (press Enter for default values):")
    
    for param_name, param_info in PARAMETER_INFO.items():
        try:
            value = st.number_input(f"{param_name} ({param_info['description']})", min_value = param_info['min_value'], max_value=param_info['max_value'], value =param_info['default'])
            params[param_name] = value
            
        except ValueError as e:
            print(f"Invalid input: {str(e)}")
    
    return params


def plot_scenario_comparison(scenarios, show_annotations):
    """Plot comparison of multiple scenarios with optional annotations."""
    # Ask user about annotations    
    if show_annotations:
        fig_width = 18
        fig = plt.figure(figsize=(fig_width, 12))
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
    else:
        fig_width = 15
        fig = plt.figure(figsize=(fig_width, 12))
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
    
    # Replace tab10 with new color generation
    colors = get_color_list(len(scenarios))
    
    def smart_annotate(ax, x, y, text, color, prev_positions=None, min_gap=0.15, side='right'):
        """Improved smart annotation placement with side selection."""
        if not show_annotations:
            return prev_positions or []
            
        if prev_positions is None:
            prev_positions = []
            
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        
        # Determine x direction based on side parameter
        x_direction = 1 if side == 'right' else -1
        x_offset = 40 if side == 'right' else -40
        y_offset = 20
        
        # Check for overlaps and adjust
        for prev_y in sorted(prev_positions):
            if abs(y - prev_y) < y_range * min_gap:
                y_offset += 30
        
        # Create background box with padding
        bbox_props = dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec=color,
            alpha=0.9,
            linewidth=0.5
        )
        
        # Add arrow properties
        arrow_props = dict(
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={0.2 if side == 'right' else -0.2}",
            color=color,
            alpha=0.6,
            linewidth=1
        )
        
        # Adjust text alignment based on side
        ha = 'left' if side == 'right' else 'right'
        
        ax.annotate(text,
                   xy=(x, y),
                   xytext=(x_offset, y_offset),
                   textcoords='offset points',
                   fontsize=9,
                   color=color,
                   bbox=bbox_props,
                   arrowprops=arrow_props,
                   ha=ha,
                   va='center')
        
        prev_positions.append(y)
        return prev_positions

    # Keep track of annotation positions
    peak_positions = []
    total_positions = []
    zero_positions = []
    units_positions = []
    
    # Plot data with improved annotations
    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        years = scenario['results']['years']
        label = scenario.get('name', f"Scenario {i+1}")
        
        # Yearly capacity plot
        capacity = scenario['results']['yearly_capacity']/1e9
        peak_capacity = max(capacity)
        peak_year = years[np.argmax(capacity)]
        ax1.plot(years, capacity, label=label, color=color)
        peak_positions = smart_annotate(ax1, peak_year, peak_capacity,
                                     f'Peak: {peak_capacity:.1f} Gt/yr',
                                     color, peak_positions,
                                     side='right' if i % 2 == 0 else 'left')
        
        # Cumulative removal plot
        cumulative = scenario['results']['cumulative_removal']/1e9
        final_removal = cumulative[-1]
        ax2.plot(years, cumulative, label=label, color=color)
        # Stagger the annotations vertically
        total_positions = smart_annotate(ax2, years[-5], final_removal,
                                      f'Total: {final_removal:.0f} Gt',
                                      color, total_positions,
                                      side='right')
        
        # Net emissions plot
        net_emissions = scenario['carbon_budget']['net_emissions']
        emission_years = scenario['carbon_budget']['years']
        ax3.plot(emission_years, net_emissions, label=label, color=color)
        
        # Find net zero crossing
        zero_cross_idx = np.where(np.diff(np.signbit(net_emissions)))[0]
        if len(zero_cross_idx) > 0:
            zero_year = emission_years[zero_cross_idx[0]]
            zero_positions = smart_annotate(ax3, zero_year, 0,
                                         f'Net Zero: {zero_year}',
                                         color, zero_positions,
                                         side='right' if i % 2 == 0 else 'left')
        
        # Active units plot
        units = scenario['results']['active_units']/1e6
        peak_units = max(units)
        peak_units_year = years[np.argmax(units)]
        ax4.plot(years, units, label=label, color=color)
        units_positions = smart_annotate(ax4, peak_units_year, peak_units,
                                      f'Peak: {peak_units:.1f}M',
                                      color, units_positions,
                                      side='right' if i % 2 == 0 else 'left')
    
    # Enhanced axis styling
    for ax, title, ylabel in [
        (ax1, 'Yearly CO₂ Removal Capacity', 'Gigatons CO₂ per Year (Gt/yr)'),
        (ax2, 'Cumulative CO₂ Removal', 'Total Gigatons CO₂ Removed (Gt)'),
        (ax3, 'Net CO₂ Emissions Trajectory', 'Gigatons CO₂ per Year (Gt/yr)'),
        (ax4, 'Active DAC Units', 'Number of Units (millions)')
    ]:
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=9)
        # Move legend outside of plot area
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Add reference lines
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    if show_annotations:
        ax1.text(2025, 1.1, '1 Gt/yr threshold', fontsize=8, alpha=0.7)
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Net Zero Line')
    ax3.fill_between(emission_years, -5, 5, alpha=0.1, color='green',
                    label='Net Zero Range')
    
    # Adjust layout to prevent cutoff
    plt.tight_layout(rect=[0, 0, 0.85 if show_annotations else 0.9, 1])
    return fig

def load_saved_scenarios(scenario_manager, n_scenarios):
    """Load and process saved scenarios."""
    scenarios = []
    available_scenarios = scenario_manager.list_scenarios()
    
    if not available_scenarios:
        st.text("No saved scenarios found. Please create new scenarios.")
        return create_new_scenarios(scenario_manager, n_scenarios)
    
    
    selected = []
    st.write("### Scenario Selection")
        
    options = [scenario[0] for scenario in available_scenarios if scenario[0] not in selected]
    selected_scenarios = st.multiselect(
        "Select scenarios:",
        options,
        key="scenario_multiselect",
        help=f"Select up to {min(n_scenarios, len(available_scenarios))} scenarios."
    )
    if st.button("Confirm Selection"):
        for name in selected_scenarios:
            try:
                params = scenario_manager.load_scenario(name)
                # Remove timestamp from params if it exists
                params.pop('timestamp', None)
                model = DACDeploymentModel(**params)
                results = model.calculate_deployment()
                scenarios.append({
                    'name': name,
                    'params': params,
                    'results': results,
                    'carbon_budget': model.calculate_carbon_budget(results)
                })
            except Exception as e:
                print(f"Error loading scenario '{name}': {str(e)}")
    return scenarios

def create_new_scenarios(scenario_manager, n_scenarios):
    """Create and process new scenarios."""
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = []
    if 'counter' not in st.session_state:
        st.session_state.counter = 1
    
    if st.session_state.counter <= n_scenarios:
        params = get_user_inputs(scenario_manager)
        model = DACDeploymentModel(**params)
        results = model.calculate_deployment()
        if st.button('Next'):
            st.session_state['scenarios'].append({
                'name': f'New Scenario {st.session_state.counter}',
                'params': params,
                'results': results,
                'carbon_budget': model.calculate_carbon_budget(results)
            })
            st.success(f"{st.session_state['scenarios'][st.session_state.counter - 1]['name']}, saved successfully!")
            st.session_state.counter = st.session_state.counter + 1

    else:
        st.write("### All Scenarios have been made")
    if n_scenarios <= st.session_state.counter:
        if st.button(f"Clear Current Scenarios {[item['name'] for item in st.session_state['scenarios']]}"):
            st.session_state['scenarios'] = []
            st.session_state.counter = 1
            
    
    return st.session_state['scenarios']

def mix_scenarios(scenario_manager, n_scenarios):
    """Mix of saved and new scenarios."""
    scenarios = []
    
    for i in range(n_scenarios):
        st.write(f"\nScenario {i+1}/{n_scenarios}")
        
        choice = st.radio("Select Option", ["Holder", "Load Saved Scenario", "Create New Scenario"])
        
        if choice == 'Load Saved Scenario':
            saved = load_saved_scenarios(scenario_manager, 1)
            if saved:
                scenarios.extend(saved)
            else:
                print("Creating new scenario instead.")
                scenarios.extend(create_new_scenarios(scenario_manager, 1))
        elif choice == "Create New Scenario":
            scenarios.extend(create_new_scenarios(scenario_manager, 1))
        else:
            st.write("Please make a decision")
    
    return scenarios

def print_scenario_comparison(scenarios):
    """Print comparison of key metrics across scenarios."""
    print("\nScenario Comparison Summary:")
    print("\nKey Metrics:")
    
    metrics = [
        ('Gigaton milestone year', lambda s: s['results']['gigaton_milestone_year']),
        ('Final yearly capacity (Gt/year)', 
         lambda s: s['results']['yearly_capacity'][-1]/1e9),
        ('Total CO2 removed (Gt)', 
         lambda s: s['results']['cumulative_removal'][-1]/1e9),
        ('Total units built', 
         lambda s: int(s['results']['cumulative_units_built'][-1])),
        ('Max atmospheric addition (Gt)', 
         lambda s: s['carbon_budget']['max_atmospheric_addition']),
        ('Probability of staying below 1.5°C', 
         lambda s: s['carbon_budget']['prob_1_5C']),
        ('Probability of staying below 2.0°C', 
         lambda s: s['carbon_budget']['prob_2_0C'])
    ]
    
    # Print header
    headers = ['Metric'] + [s.get('name', f"Scenario {i+1}") 
                           for i, s in enumerate(scenarios)]
    print('\n' + '  '.join(f"{h:<30}" for h in headers))
    print('-' * (30 * len(headers)))
    
    # Print each metric
    for metric_name, metric_func in metrics:
        row = [metric_name]
        for scenario in scenarios:
            try:
                value = metric_func(scenario)
                if isinstance(value, (int, np.integer)):
                    row.append(f"{value:,}")
                else:
                    row.append(f"{value:.2f}")
            except Exception as e:
                row.append("N/A")
        print('  '.join(f"{v:<30}" for v in row))

def print_monte_carlo_summary(summary_stats, correlation_matrix):
    """Print summary statistics and key correlations with proper column handling."""
    print("\nMonte Carlo Analysis Summary:")
    for metric, stats in summary_stats.items():
        print(f"\n{metric}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.2f}")
    
    print("\nKey Parameter Correlations:")
    param_cols = [col for col in correlation_matrix.index if col.startswith('Param_')]
    
    # Use exact column names from DataFrame
    target_metrics = ['Max Atmospheric Addition (Gt)', 'Total Removal by 2075 (Gt)']
    
    for param in param_cols:
        for metric in target_metrics:
            try:
                if metric in correlation_matrix.columns:
                    corr = correlation_matrix.loc[param, metric]
                    print(f"{param} vs {metric}: {corr:.3f}")
            except Exception as e:
                print(f"Could not calculate correlation for {param} vs {metric}")

def save_and_plot_results(results_df, summary_stats, correlation_matrix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results to CSV
    results_df.to_csv(f'monte_carlo_results_{timestamp}.csv')
    
    # Generate and display plots
    figs = plot_monte_carlo_results(results_df, summary_stats, correlation_matrix)
    
    for i, fig in enumerate(figs):
        # Save plot
        fig.savefig(f'monte_carlo_plot_{i}_{timestamp}.png', dpi=300, bbox_inches='tight')
        # Display plot
        plt.figure(fig.number)
        st.pyplot(fig)
        plt.close(fig)
    
    print(f"\nResults saved with timestamp: {timestamp}")
    print(f"Results file: monte_carlo_results_{timestamp}.csv")
    print(f"Plot files: monte_carlo_plot_[0-{len(figs)-1}]_{timestamp}.png")

def plot_levelized_cost_distribution(cost_results, ax):
    """Plot distribution of levelized costs."""
    costs = [result for result in cost_results['levelized_cost']]
    sns.histplot(data=costs, bins=30, ax=ax)
    median = np.median(costs)
    ax.axvline(median, color='r', linestyle='-', 
               label=f'Median: ${median:.0f}/tCO2')
    ax.set_title('Levelized Cost Distribution')
    ax.set_xlabel('Cost ($/tCO2)')
    ax.legend()

def plot_learning_curve_savings(cost_results, ax):
    """Plot cumulative savings from learning curve effects."""
    savings = cost_results['learning_curve_savings']
    ax.plot(np.cumsum(savings), 'b-')
    ax.set_title('Cumulative Learning Curve Savings')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Savings ($)')
    ax.grid(True)

def plot_cost_breakdown(cost_results, ax):
    """Plot breakdown of CAPEX vs OPEX."""
    breakdowns = cost_results['cost_breakdown']
    capex = [b['capex'] for b in breakdowns]
    opex = [b['opex'] for b in breakdowns]
    
    ax.boxplot([capex, opex], labels=['CAPEX', 'OPEX'])
    ax.set_title('Cost Component Distribution')
    ax.set_ylabel('Cost ($)')
    ax.grid(True)

def plot_investment_timeline(cost_results, ax):
    """Plot yearly investment requirements."""
    yearly_costs = np.mean(cost_results['yearly_costs'], axis=0)
    years = np.arange(len(yearly_costs))
    ax.bar(years, yearly_costs/1e9)
    ax.set_title('Annual Investment Requirements')
    ax.set_xlabel('Year')
    ax.set_ylabel('Investment (Billion $)')
    ax.grid(True)

def plot_buildout_rates(timeline_results, ax):
    """Plot distribution of buildout rates."""
    rates = [r['max_rate'] for r in timeline_results['buildout_rates']]
    sns.histplot(data=rates, bins=30, ax=ax)
    ax.set_title('Distribution of Maximum Buildout Rates')
    ax.set_xlabel('Units per Year')
    ax.grid(True)

def plot_supply_chain_requirements(timeline_results, ax):
    """Plot key supply chain requirements with improved scaling."""
    if not timeline_results['supply_chain_requirements']:
        return
        
    total_reqs = defaultdict(float)
    for req in timeline_results['supply_chain_requirements']:
        for material, amount in req.items():
            total_reqs[material] += amount
    
    n_sims = len(timeline_results['supply_chain_requirements'])
    avg_reqs = {k: v/n_sims for k, v in total_reqs.items()}
    
    materials = list(avg_reqs.keys())
    values = [avg_reqs[m] for m in materials]
    
    # Format values for better readability
    formatted_values = []
    units = {'steel': 'ktons', 'sorbent': 'tons', 'electricity': 'GWh'}
    for m, v in zip(materials, values):
        if m == 'steel':
            v = v / 1000  # Convert to ktons
        elif m == 'electricity':
            v = v / 1000  # Convert to GWh
        formatted_values.append(v)
    
    bars = ax.bar(materials, formatted_values)
    ax.set_title('Annual Supply Chain Requirements')
    ax.set_ylabel('Amount Required')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

def plot_critical_path(timeline_results, ax):
    """Plot critical path timeline with improved formatting."""
    if not timeline_results['critical_path_metrics']:
        return
        
    avg_metrics = defaultdict(float)
    n_metrics = len(timeline_results['critical_path_metrics'])
    
    for metric in timeline_results['critical_path_metrics']:
        for key, value in metric.items():
            avg_metrics[key] += value / n_metrics
    
    items = list(avg_metrics.keys())
    values = [avg_metrics[item] for item in items]
    
    # Format labels for readability
    labels = [item.replace('_', ' ').title() for item in items]
    
    bars = ax.bar(labels, values)
    ax.set_title('Critical Path Timeline')
    ax.set_ylabel('Time (Years)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2)

def plot_manufacturing_capacity(timeline_results, ax):
    """Plot required manufacturing capacity over time."""
    factory_data = timeline_results['manufacturing_capacity']
    if not factory_data:
        return
        
    # Get first timeline length
    first_timeline = factory_data[0]['factory_timeline']
    timeline_length = len(first_timeline)
    years = np.arange(timeline_length)
    
    # Stack all timelines
    all_timelines = np.zeros((len(factory_data), timeline_length))
    for i, data in enumerate(factory_data):
        timeline = data['factory_timeline']
        if len(timeline) == timeline_length:
            # Ensure non-negative values
            all_timelines[i] = np.maximum(timeline, 0)
    
    # Calculate mean capacity
    mean_capacity = np.mean(all_timelines, axis=0)
    
    # Plot
    ax.plot(years, mean_capacity, 'b-')
    ax.set_title('Required Manufacturing Capacity')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Factories')
    ax.grid(True)
    
    # Set reasonable y-axis limits
    ax.set_ylim(bottom=0, top=np.ceil(np.max(mean_capacity) * 1.1))

def calculate_material_requirements(yearly_buildout):
    """Calculate material requirements based on buildout rate."""
    return {
        'steel': np.maximum(yearly_buildout * 1000, 0).sum(),  # tons
        'sorbent': np.maximum(yearly_buildout * 100, 0).sum(),  # tons
        'electricity': np.maximum(yearly_buildout * 8760 * 0.5, 0).sum()  # MWh
    }

def run_selected_analysis(choice):
           
    """Run the selected advanced analysis option."""
    if choice == "Cost Analysis":
        # Cost Analysis
        results_df, _, _ = run_monte_carlo(n_simulations=100)
        cost_results = run_cost_analysis(results_df)
        fig = plot_cost_analysis(cost_results)
        st.pyplot(fig)
        return cost_results
        
    elif choice == "Timeline Analysis":
        # Timeline Analysis
        results_df, _, _ = run_monte_carlo(n_simulations=100)
        timeline_results = run_timeline_analysis(results_df)
        fig = plot_timeline_analysis(timeline_results)
        st.pyplot(fig)
        return timeline_results
        
    elif choice == "Sensitivity Analysis":
        # Sensitivity Analysis
        results_df, _, correlation_matrix = run_monte_carlo(n_simulations=500)
        fig = plot_sensitivity_analysis(results_df)
        st.pyplot(fig)
        return correlation_matrix
        
    elif choice ==  "Regional Analysis":
        # Regional Analysis (placeholder)
        st.write("Regional analysis not implemented yet")
        return None
    
    else:
        print("Invalid choice")
        return None
    
def run_advanced_analysis():
    st.subheader("Advanced Analysis Options")
    
    # Define the radio button for analysis selection
    choice = st.radio(
        "Select an option for analysis:",
        options=[
            "Cost Analysis",
            "Timeline Analysis",
            "Sensitivity Analysis",
            "Regional Analysis"
        ],
        index=0  # Default to the first option, or omit for no default selection
    )
    
    return run_selected_analysis(choice)

def plot_timeline_analysis(timeline_results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    plot_buildout_rates(timeline_results, ax1)
    plot_manufacturing_capacity(timeline_results, ax2)
    plot_supply_chain_requirements(timeline_results, ax3)
    plot_critical_path(timeline_results, ax4)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":

    # Streamlit App
    st.set_page_config(layout="wide")
    st.title("DAC Deployment Model with Monte Carlo Analysis")

    # Sidebar for navigation
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode:",
        (
            "Single Scenario Analysis",
            "Multiple Scenario Comparison",
            "Monte Carlo Analysis",
            "Target-based Monte Carlo Analysis",
            "Advanced Analysis",
            "Exit",
        )
    )

    # Main Section
    if analysis_mode == "Single Scenario Analysis":
        st.header("Single Scenario Analysis")

        with st.form(key="single_scenario"):
            
            # Collect inputs for single scenario
            st.write("Provide inputs for the single scenario.")
            scenario_manager = ScenarioManager()
            params = get_user_inputs(scenario_manager)
            
            # Replace this with your input collection logic            
            submit = st.form_submit_button("Run Analysis")
            saved = st.text_input("Save Analysis (Leave Blank if you do not want to save Analysis)")
            if submit:
                try:
                    model = DACDeploymentModel(**params)
                    results = model.calculate_deployment()
                    #st.write(results)
                    fig = model.plot_results()
                    st.pyplot(fig)
                    if (saved != ""):
                        scenario_manager.save_scenario(saved, params)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        

    elif analysis_mode == "Multiple Scenario Comparison":
        st.header("Multiple Scenario Comparison")
        n_scenarios = st.number_input("Number of scenarios to compare:", min_value=1, max_value=3, value=2)
        if 'scenarios' not in st.session_state:
            st.session_state.scenarios = {}
        scenarios = st.session_state.scenarios
        st.session_state.scenarios = run_multiple_scenarios(n_scenarios)

        run_comparison = st.button("Run Comparison")

        show_annotations = st.checkbox("Show Annotations (must re-confirm selection after)?")
        if run_comparison:
            try:

                if len(scenarios) > 0:
                    st.write("Scenario Comparison Results:")
                    
                    fig = plot_scenario_comparison(scenarios, show_annotations)
                    st.pyplot(fig)
                else:
                    st.warning("No scenarios to compare.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif analysis_mode == "Monte Carlo Analysis":
        st.header("Monte Carlo Analysis")
        n_sims = st.number_input("Number of Monte Carlo simulations:", min_value=100, max_value=10000, value=1000)
        run_analysis = st.button("Run Monte Carlo Analysis")
        if run_analysis:
            try:
                results_df, summary_stats, correlation_matrix = run_monte_carlo(n_simulations=n_sims)
                st.write("Monte Carlo Summary Statistics:")
                st.write(summary_stats)
                st.write("Correlation Matrix:")
                st.write(correlation_matrix)
                save_and_plot_results(results_df, summary_stats, correlation_matrix)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif analysis_mode == "Target-based Monte Carlo Analysis":
        st.header("Target-based Monte Carlo Analysis")
        
        
        try:
            run_target_based_analysis()
        except Exception as e:
            st.error(f"Error: {str(e)}")

    elif analysis_mode == "Advanced Analysis":
        st.header("Advanced Analysis")
        run_advanced_analysis()
        