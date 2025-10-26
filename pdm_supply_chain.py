# pdm_supply_chain.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PDMSupplyChainOptimizer:
    """
    Supply chain optimizer for PDM (Predictive Data Maintenance) system.
    Optimizes spare parts inventory based on predicted machine failures.
    """
    
    def __init__(self):
        self.warehouses = {}
        self.machine_locations = {}
        self.shipping_costs = {}
        self.transit_times = {}
        self.component_costs = {}
        self.failure_rates = {}
        
    def setup_warehouses(self, warehouse_config: Dict):
        """
        Setup warehouse locations and capacities.
        
        Args:
            warehouse_config: Dict with warehouse info
            Example: {
                'singapore': {'capacity': 1000, 'location': (1.3521, 103.8198), 'cost_per_unit': 50},
                'tokyo': {'capacity': 800, 'location': (35.6762, 139.6503), 'cost_per_unit': 60},
                'sydney': {'capacity': 600, 'location': (-33.8688, 151.2093), 'cost_per_unit': 55}
            }
        """
        self.warehouses = warehouse_config
        
    def setup_machines(self, machine_config: Dict):
        """
        Setup machine locations and failure patterns.
        
        Args:
            machine_config: Dict with machine info
            Example: {
                1: {'location': (1.2966, 103.7764), 'region': 'singapore', 'criticality': 'high'},
                2: {'location': (35.6762, 139.6503), 'region': 'tokyo', 'criticality': 'medium'}
            }
        """
        self.machine_locations = machine_config
        
    def setup_components(self, component_config: Dict):
        """
        Setup component information and costs.
        
        Args:
            component_config: Dict with component info
            Example: {
                'comp1': {'cost': 1000, 'lead_time_days': 7, 'criticality': 'high'},
                'comp2': {'cost': 500, 'lead_time_days': 5, 'criticality': 'medium'},
                'comp3': {'cost': 200, 'lead_time_days': 3, 'criticality': 'low'},
                'comp4': {'cost': 1500, 'lead_time_days': 10, 'criticality': 'high'}
            }
        """
        self.component_costs = component_config
        
    def calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """
        Calculate approximate distance between two locations using Haversine formula.
        """
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Earth's radius in kilometers
        
        return c * r
    
    def calculate_shipping_cost(self, warehouse: str, machine_id: int, component: str, quantity: int) -> float:
        """
        Calculate shipping cost from warehouse to machine.
        """
        if warehouse not in self.warehouses or machine_id not in self.machine_locations:
            return float('inf')
        
        warehouse_loc = self.warehouses[warehouse]['location']
        machine_loc = self.machine_locations[machine_id]['location']
        
        distance = self.calculate_distance(warehouse_loc, machine_loc)
        
        # Base shipping cost per km
        base_cost_per_km = 0.5
        
        # Component-specific multiplier
        component_cost = self.component_costs.get(component, {}).get('cost', 1000)
        cost_multiplier = 1 + (component_cost / 10000)  # Higher cost components have higher shipping multiplier
        
        # Urgency multiplier
        machine_criticality = self.machine_locations[machine_id].get('criticality', 'medium')
        urgency_multiplier = {'high': 1.5, 'medium': 1.0, 'low': 0.8}.get(machine_criticality, 1.0)
        
        total_cost = distance * base_cost_per_km * cost_multiplier * urgency_multiplier * quantity
        
        return total_cost
    
    def optimize_inventory_allocation(self, failure_predictions: Dict, 
                                    current_inventory: Dict, 
                                    budget_constraint: float = 100000,
                                    time_horizon_days: int = 30) -> Dict:
        """
        Optimize spare parts allocation across warehouses.
        
        Args:
            failure_predictions: Dict with predicted failures by machine and component
            current_inventory: Dict with current inventory levels
            budget_constraint: Maximum budget for inventory allocation
            time_horizon_days: Planning horizon in days
        
        Returns:
            Dict with optimized allocation plan
        """
        print("🔄 Optimizing inventory allocation...")
        
        # Extract unique warehouses and components
        warehouses = list(self.warehouses.keys())
        components = list(self.component_costs.keys())
        
        # Create optimization variables: inventory allocation for each warehouse-component pair
        n_vars = len(warehouses) * len(components)
        
        # Create variable mapping
        var_mapping = {}
        idx = 0
        for warehouse in warehouses:
            for component in components:
                var_mapping[(warehouse, component)] = idx
                idx += 1
        
        # Objective function: minimize total cost while meeting demand
        def objective(x):
            total_cost = 0
            
            # Inventory holding cost
            for warehouse in warehouses:
                for component in components:
                    var_idx = var_mapping[(warehouse, component)]
                    quantity = x[var_idx]
                    holding_cost_per_unit = self.warehouses[warehouse].get('cost_per_unit', 50)
                    component_cost = self.component_costs[component].get('cost', 1000)
                    total_cost += quantity * holding_cost_per_unit * (time_horizon_days / 30)  # Monthly cost
                    total_cost += quantity * component_cost  # Component cost
            
            return total_cost
        
        # Constraints
        constraints = []
        
        # Budget constraint
        def budget_constraint_func(x):
            total_cost = 0
            for warehouse in warehouses:
                for component in components:
                    var_idx = var_mapping[(warehouse, component)]
                    quantity = x[var_idx]
                    component_cost = self.component_costs[component].get('cost', 1000)
                    total_cost += quantity * component_cost
            return budget_constraint - total_cost
        
        constraints.append({'type': 'ineq', 'fun': budget_constraint_func})
        
        # Capacity constraints for each warehouse
        for warehouse in warehouses:
            def make_capacity_constraint(wh):
                def capacity_constraint(x):
                    total_inventory = 0
                    for component in components:
                        var_idx = var_mapping[(wh, component)]
                        total_inventory += x[var_idx]
                    return self.warehouses[wh]['capacity'] - total_inventory
                return capacity_constraint
            constraints.append({'type': 'ineq', 'fun': make_capacity_constraint(warehouse)})
        
        # Demand satisfaction constraints for each component
        for component in components:
            def make_demand_constraint(comp):
                def demand_constraint(x):
                    total_available = 0
                    for warehouse in warehouses:
                        var_idx = var_mapping[(warehouse, comp)]
                        total_available += x[var_idx]
                    
                    # Calculate total demand for this component
                    total_demand = 0
                    for machine_id, predictions in failure_predictions.items():
                        if isinstance(predictions, dict) and comp in predictions:
                            total_demand += predictions[comp]
                        elif isinstance(predictions, (int, float)):
                            # If predictions is a single number, assume it's for this component
                            total_demand += predictions
                    
                    return total_available - total_demand
                return demand_constraint
            constraints.append({'type': 'ineq', 'fun': make_demand_constraint(component)})
        
        # Bounds: non-negative inventory
        bounds = [(0, None) for _ in range(n_vars)]
        
        # Initial guess
        x0 = np.ones(n_vars) * 10
        
        # Solve optimization
        result = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        # Create allocation plan
        allocation_plan = {}
        for warehouse in warehouses:
            allocation_plan[warehouse] = {
                'total_capacity': self.warehouses[warehouse]['capacity'],
                'components': {}
            }
            
            total_inventory = 0
            for component in components:
                var_idx = var_mapping[(warehouse, component)]
                recommended_quantity = int(result.x[var_idx])
                current_quantity = current_inventory.get(warehouse, {}).get(component, 0)
                
                allocation_plan[warehouse]['components'][component] = {
                    'recommended_quantity': recommended_quantity,
                    'current_quantity': current_quantity,
                    'additional_needed': max(0, recommended_quantity - current_quantity),
                    'component_cost': self.component_costs[component].get('cost', 1000),
                    'lead_time_days': self.component_costs[component].get('lead_time_days', 5)
                }
                
                total_inventory += recommended_quantity
            
            allocation_plan[warehouse]['total_recommended'] = total_inventory
            allocation_plan[warehouse]['utilization'] = total_inventory / self.warehouses[warehouse]['capacity']
        
        return {
            'allocation_plan': allocation_plan,
            'total_cost': result.fun,
            'optimization_success': result.success,
            'message': result.message,
            'budget_used': budget_constraint - result.fun
        }
    
    def generate_recommendations(self, allocation_plan: Dict) -> List[str]:
        """
        Generate actionable recommendations based on allocation plan.
        """
        recommendations = []
        
        for warehouse, plan in allocation_plan['allocation_plan'].items():
            warehouse_recommendations = []
            
            for component, comp_plan in plan['components'].items():
                if comp_plan['additional_needed'] > 0:
                    warehouse_recommendations.append(
                        f"📦 Order {comp_plan['additional_needed']} units of {component} "
                        f"(lead time: {comp_plan['lead_time_days']} days, "
                        f"cost: ${comp_plan['component_cost']:,} each)"
                    )
                elif comp_plan['current_quantity'] > comp_plan['recommended_quantity'] * 1.2:
                    warehouse_recommendations.append(
                        f"📉 Consider reducing {component} inventory at {warehouse.title()} "
                        f"(currently {comp_plan['current_quantity']}, recommended {comp_plan['recommended_quantity']})"
                    )
            
            if warehouse_recommendations:
                recommendations.append(f"\n🏢 {warehouse.title()} Warehouse:")
                recommendations.extend(warehouse_recommendations)
            
            if plan['utilization'] > 0.9:
                recommendations.append(
                    f"⚠️ High utilization at {warehouse.title()} warehouse ({plan['utilization']:.1%})"
                )
        
        return recommendations
    
    def calculate_business_impact(self, allocation_plan: Dict, failure_predictions: Dict) -> Dict:
        """
        Calculate business impact of the allocation plan.
        """
        # Calculate total predicted failures
        total_failures = 0
        for machine_id, predictions in failure_predictions.items():
            if isinstance(predictions, dict):
                total_failures += sum(predictions.values())
            elif isinstance(predictions, (int, float)):
                total_failures += predictions
        
        # Calculate total recommended inventory
        total_inventory = 0
        for warehouse, plan in allocation_plan['allocation_plan'].items():
            total_inventory += plan['total_recommended']
        
        # Calculate cost savings
        emergency_shipping_cost = 2000  # Cost per emergency shipment
        regular_shipping_cost = 200     # Cost per regular shipment
        downtime_cost_per_hour = 5000   # Cost per hour of downtime
        
        emergency_shipments_avoided = max(0, total_failures - total_inventory)
        shipping_cost_savings = emergency_shipments_avoided * (emergency_shipping_cost - regular_shipping_cost)
        downtime_reduction_hours = emergency_shipments_avoided * 24  # Assume 24 hours saved per avoided emergency
        downtime_cost_savings = downtime_reduction_hours * downtime_cost_per_hour
        
        # Calculate inventory cost
        inventory_cost = 0
        for warehouse, plan in allocation_plan['allocation_plan'].items():
            for component, comp_plan in plan['components'].items():
                inventory_cost += comp_plan['recommended_quantity'] * comp_plan['component_cost']
        
        net_savings = shipping_cost_savings + downtime_cost_savings - inventory_cost
        
        return {
            'total_predicted_failures': total_failures,
            'total_recommended_inventory': total_inventory,
            'emergency_shipments_avoided': emergency_shipments_avoided,
            'shipping_cost_savings': shipping_cost_savings,
            'downtime_cost_savings': downtime_cost_savings,
            'inventory_cost': inventory_cost,
            'net_savings': net_savings,
            'roi_percentage': (net_savings / inventory_cost * 100) if inventory_cost > 0 else 0
        }
    
    def visualize_allocation(self, allocation_plan: Dict, save_path: str = None):
        """
        Create visualization of the allocation plan.
        """
        warehouses = list(allocation_plan['allocation_plan'].keys())
        components = list(allocation_plan['allocation_plan'][warehouses[0]]['components'].keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Inventory by warehouse
        warehouse_totals = [allocation_plan['allocation_plan'][w]['total_recommended'] for w in warehouses]
        axes[0, 0].bar(warehouses, warehouse_totals, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Total Inventory by Warehouse')
        axes[0, 0].set_ylabel('Units')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Component distribution
        component_totals = {}
        for warehouse in warehouses:
            for component in components:
                if component not in component_totals:
                    component_totals[component] = 0
                component_totals[component] += allocation_plan['allocation_plan'][warehouse]['components'][component]['recommended_quantity']
        
        axes[0, 1].bar(component_totals.keys(), component_totals.values(), alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Total Inventory by Component')
        axes[0, 1].set_ylabel('Units')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Warehouse utilization
        utilizations = [allocation_plan['allocation_plan'][w]['utilization'] for w in warehouses]
        colors = ['red' if u > 0.9 else 'orange' if u > 0.7 else 'green' for u in utilizations]
        
        axes[1, 0].bar(warehouses, utilizations, color=colors, alpha=0.7)
        axes[1, 0].set_title('Warehouse Utilization Rates')
        axes[1, 0].set_ylabel('Utilization Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add utilization percentage labels
        for i, v in enumerate(utilizations):
            axes[1, 0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        # 4. Component cost analysis
        component_costs = []
        component_names = []
        for component in components:
            total_cost = 0
            for warehouse in warehouses:
                qty = allocation_plan['allocation_plan'][warehouse]['components'][component]['recommended_quantity']
                cost_per_unit = allocation_plan['allocation_plan'][warehouse]['components'][component]['component_cost']
                total_cost += qty * cost_per_unit
            component_costs.append(total_cost)
            component_names.append(component)
        
        axes[1, 1].pie(component_costs, labels=component_names, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Inventory Cost Distribution by Component')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

def create_sample_pdm_supply_chain():
    """
    Create sample PDM supply chain configuration.
    """
    optimizer = PDMSupplyChainOptimizer()
    
    # Setup warehouses
    warehouse_config = {
        'singapore': {'capacity': 1000, 'location': (1.3521, 103.8198), 'cost_per_unit': 50},
        'tokyo': {'capacity': 800, 'location': (35.6762, 139.6503), 'cost_per_unit': 60},
        'sydney': {'capacity': 600, 'location': (-33.8688, 151.2093), 'cost_per_unit': 55}
    }
    optimizer.setup_warehouses(warehouse_config)
    
    # Setup machines
    machine_config = {
        1: {'location': (1.2966, 103.7764), 'region': 'singapore', 'criticality': 'high'},
        2: {'location': (35.6762, 139.6503), 'region': 'tokyo', 'criticality': 'medium'},
        3: {'location': (-33.8688, 151.2093), 'region': 'sydney', 'criticality': 'high'},
        4: {'location': (1.3521, 103.8198), 'region': 'singapore', 'criticality': 'low'},
        5: {'location': (35.6762, 139.6503), 'region': 'tokyo', 'criticality': 'medium'}
    }
    optimizer.setup_machines(machine_config)
    
    # Setup components
    component_config = {
        'comp1': {'cost': 1000, 'lead_time_days': 7, 'criticality': 'high'},
        'comp2': {'cost': 500, 'lead_time_days': 5, 'criticality': 'medium'},
        'comp3': {'cost': 200, 'lead_time_days': 3, 'criticality': 'low'},
        'comp4': {'cost': 1500, 'lead_time_days': 10, 'criticality': 'high'}
    }
    optimizer.setup_components(component_config)
    
    return optimizer

if __name__ == '__main__':
    # Test the PDM supply chain optimizer
    print("🔄 Testing PDM Supply Chain Optimizer...")
    
    # Create optimizer
    optimizer = create_sample_pdm_supply_chain()
    
    # Sample failure predictions (from ML models)
    failure_predictions = {
        1: {'comp1': 2, 'comp2': 1, 'comp3': 3, 'comp4': 1},  # Machine 1 predictions
        2: {'comp1': 1, 'comp2': 2, 'comp3': 2, 'comp4': 0},  # Machine 2 predictions
        3: {'comp1': 3, 'comp2': 1, 'comp3': 1, 'comp4': 2},  # Machine 3 predictions
        4: {'comp1': 0, 'comp2': 1, 'comp3': 2, 'comp4': 0},  # Machine 4 predictions
        5: {'comp1': 1, 'comp2': 2, 'comp3': 1, 'comp4': 1}   # Machine 5 predictions
    }
    
    # Current inventory levels
    current_inventory = {
        'singapore': {'comp1': 5, 'comp2': 10, 'comp3': 20, 'comp4': 3},
        'tokyo': {'comp1': 3, 'comp2': 8, 'comp3': 15, 'comp4': 2},
        'sydney': {'comp1': 2, 'comp2': 5, 'comp3': 10, 'comp4': 1}
    }
    
    # Optimize allocation
    allocation_plan = optimizer.optimize_inventory_allocation(
        failure_predictions, 
        current_inventory, 
        budget_constraint=50000
    )
    
    print("\n=== ALLOCATION PLAN ===")
    for warehouse, plan in allocation_plan['allocation_plan'].items():
        print(f"\n🏢 {warehouse.title()} Warehouse:")
        print(f"   Total Recommended: {plan['total_recommended']} units")
        print(f"   Utilization: {plan['utilization']:.1%}")
        print(f"   Components:")
        for component, comp_plan in plan['components'].items():
            print(f"     {component}: {comp_plan['recommended_quantity']} units "
                  f"(current: {comp_plan['current_quantity']}, "
                  f"needed: {comp_plan['additional_needed']})")
    
    print(f"\n💰 Total Cost: ${allocation_plan['total_cost']:,.2f}")
    print(f"✅ Optimization Success: {allocation_plan['optimization_success']}")
    
    # Generate recommendations
    recommendations = optimizer.generate_recommendations(allocation_plan)
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in recommendations:
        print(rec)
    
    # Calculate business impact
    business_impact = optimizer.calculate_business_impact(allocation_plan, failure_predictions)
    print(f"\n📊 BUSINESS IMPACT:")
    print(f"   Predicted Failures: {business_impact['total_predicted_failures']}")
    print(f"   Recommended Inventory: {business_impact['total_recommended_inventory']}")
    print(f"   Emergency Shipments Avoided: {business_impact['emergency_shipments_avoided']}")
    print(f"   Net Savings: ${business_impact['net_savings']:,.2f}")
    print(f"   ROI: {business_impact['roi_percentage']:.1f}%")
    
    # Create visualization
    optimizer.visualize_allocation(allocation_plan, 'pdm_inventory_allocation.png')
    
    print("\n✅ PDM Supply Chain Optimizer test completed!")
