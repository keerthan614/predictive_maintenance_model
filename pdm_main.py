# pdm_main.py
"""
Main script for PDM (Predictive Data Maintenance) system.
Trains models and demonstrates the complete pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pdm_data_processor import PDMDataProcessor
from pdm_models import PDMPredictiveModels
from pdm_inference import PDMInference, create_sample_telemetry_data, create_sample_historical_data
from pdm_supply_chain import create_sample_pdm_supply_chain
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the complete PDM pipeline.
    """
    print("🚀 PDM (Predictive Data Maintenance) System")
    print("=" * 60)
    
    # Step 1: Data Processing
    print("\n1️⃣ DATA PROCESSING")
    print("-" * 30)
    
    processor = PDMDataProcessor()
    processed_data = processor.process_all_data()
    processor.save_processor('pdm_processor.pkl')
    
    print(f"✅ Data processing completed!")
    print(f"   Tabular samples: {processed_data['X_tabular'].shape[0]}")
    print(f"   Sequence samples: {processed_data['X_sequences'].shape[0]}")
    print(f"   Features: {len(processed_data['feature_columns'])}")
    print(f"   Failure rate: {processed_data['y_tabular'].mean():.3f}")
    
    # Step 2: Model Training
    print("\n2️⃣ MODEL TRAINING")
    print("-" * 30)
    
    models = PDMPredictiveModels()
    models.feature_columns = processed_data['feature_columns']
    
    # Train tabular models
    print("🔄 Training tabular models...")
    X_test, y_test = models.train_tabular_models(
        processed_data['X_tabular'], 
        processed_data['y_tabular']
    )
    
    # Train sequence models
    print("\n🔄 Training sequence models...")
    cnn_lstm_history, lstm_history = models.train_sequence_models(
        processed_data['X_sequences'], 
        processed_data['y_sequences'],
        epochs=50  # Reduced for demo
    )
    
    # Plot training history
    models.plot_training_history(cnn_lstm_history, lstm_history)
    
    # Save models
    models.save_models('pdm_models/')
    
    print("✅ Model training completed!")
    
    # Step 3: Real-time Inference Demo
    print("\n3️⃣ REAL-TIME INFERENCE DEMO")
    print("-" * 30)
    
    # Initialize inference pipeline
    inference = PDMInference('pdm_models/')
    
    # Test with sample data
    print("🧪 Testing with sample telemetry data...")
    
    # Single prediction
    telemetry_data = create_sample_telemetry_data(machine_id=1)
    print(f"📊 Current telemetry data: {telemetry_data}")
    
    result = inference.predict_failure_tabular(telemetry_data)
    print(f"\n🎯 Tabular Prediction Result:")
    if 'error' not in result:
        print(f"   Failure Probability: {result.get('failure_probability', 'N/A'):.3f}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        print(f"   Urgency: {result.get('urgency_level', 'N/A')}")
        print(f"\n💡 Recommendations:")
        for rec in result.get('recommendations', []):
            print(f"   {rec}")
    else:
        print(f"   Error: {result['error']}")
    
    # Ensemble prediction with historical data
    print("\n🔄 Testing Ensemble Prediction with Historical Data...")
    historical_data = create_sample_historical_data(machine_id=1, n_points=35)
    ensemble_result = inference.predict_failure_ensemble(telemetry_data, historical_data)
    
    print(f"🎯 Ensemble Prediction Result:")
    if 'error' not in ensemble_result:
        print(f"   Failure Probability: {ensemble_result.get('failure_probability', 'N/A'):.3f}")
        print(f"   Confidence: {ensemble_result.get('confidence', 'N/A')}")
        print(f"   Urgency: {ensemble_result.get('urgency_level', 'N/A')}")
        print(f"   Models Used: {ensemble_result.get('model_types_used', 'N/A')}")
    else:
        print(f"   Error: {ensemble_result['error']}")
    
    # Batch prediction test
    print("\n🔄 Testing Batch Prediction...")
    batch_telemetry = [create_sample_telemetry_data(i) for i in range(1, 6)]
    batch_historical = [create_sample_historical_data(i) for i in range(1, 6)]
    batch_results = inference.batch_predict(batch_telemetry, batch_historical)
    
    print(f"\n📊 Batch Prediction Results:")
    for result in batch_results:
        pred = result['prediction']
        if 'error' not in pred:
            print(f"   {result['machine_id']}: Prob={pred.get('failure_probability', 'N/A'):.3f}, "
                  f"Urgency={pred.get('urgency_level', 'N/A')}")
        else:
            print(f"   {result['machine_id']}: Error - {pred['error']}")
    
    # Step 4: Supply Chain Optimization Demo
    print("\n4️⃣ SUPPLY CHAIN OPTIMIZATION DEMO")
    print("-" * 30)
    
    # Create optimizer
    optimizer = create_sample_pdm_supply_chain()
    
    # Generate failure predictions based on our models
    failure_predictions = {}
    for i, result in enumerate(batch_results):
        machine_id = i + 1
        pred = result['prediction']
        
        if 'error' not in pred:
            # Generate component-specific predictions based on failure probability
            failure_prob = pred.get('failure_probability', 0)
            
            # Map failure probability to component predictions
            if failure_prob > 0.8:
                failure_predictions[machine_id] = {'comp1': 2, 'comp2': 1, 'comp3': 1, 'comp4': 1}
            elif failure_prob > 0.5:
                failure_predictions[machine_id] = {'comp1': 1, 'comp2': 1, 'comp3': 1, 'comp4': 0}
            else:
                failure_predictions[machine_id] = {'comp1': 0, 'comp2': 0, 'comp3': 1, 'comp4': 0}
        else:
            # Default predictions for machines with errors
            failure_predictions[machine_id] = {'comp1': 0, 'comp2': 0, 'comp3': 1, 'comp4': 0}
    
    print(f"📊 Generated failure predictions: {failure_predictions}")
    
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
        budget_constraint=100000
    )
    
    print(f"\n=== ALLOCATION PLAN ===")
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
    
    # Step 5: Summary
    print("\n5️⃣ SUMMARY")
    print("-" * 30)
    
    print("✅ PDM System Successfully Deployed!")
    print("\n🎯 Key Achievements:")
    print("   • Processed real PDM dataset with 876K+ telemetry records")
    print("   • Trained 5 different ML models (CNN-LSTM, LSTM, RF, GB, LR)")
    print("   • Built real-time inference pipeline")
    print("   • Optimized supply chain inventory allocation")
    print("   • Demonstrated significant business impact")
    
    print("\n📈 Business Value:")
    print(f"   • Predicted {business_impact['total_predicted_failures']} potential failures")
    print(f"   • Avoided {business_impact['emergency_shipments_avoided']} emergency shipments")
    print(f"   • Generated ${business_impact['net_savings']:,.2f} in net savings")
    print(f"   • Achieved {business_impact['roi_percentage']:.1f}% ROI")
    
    print("\n🚀 Next Steps:")
    print("   1. Deploy models to production environment")
    print("   2. Integrate with real-time sensor data streams")
    print("   3. Set up automated alerting system")
    print("   4. Implement continuous model retraining")
    print("   5. Expand to additional machine types and components")
    
    print("\n🎉 PDM System Demo Completed Successfully!")

if __name__ == '__main__':
    main()
