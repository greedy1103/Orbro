graph TD
    A["❓ WHY are these methods needed?"] --> B["🎯 PROBLEM: No Ground Truth"]
    
    B --> C["🔬 SCIENTIFIC FOUNDATIONS"]
    
    C --> D["📊 Cross-Validation<br/>🎓 Theory: Machine Learning Theory<br/>📚 Source: Hastie et al. (2009)<br/>✅ Why: Estimate generalization error"]
    
    C --> E["🧮 Statistical Significance<br/>🎓 Theory: Statistical Hypothesis Testing<br/>📚 Source: Student's t-test (1908)<br/>✅ Why: Remove randomness effects"]
    
    C --> F["🛡️ Robustness Test<br/>🎓 Theory: Robust Statistics<br/>📚 Source: Huber (1981)<br/>✅ Why: Test stability under perturbations"]
    
    C --> G["📏 Confidence Intervals<br/>🎓 Theory: Statistical Inference<br/>📚 Source: Neyman (1937)<br/>✅ Why: Quantify uncertainty"]
    
    C --> H["🎯 Combined Scoring<br/>🎓 Theory: Multi-criteria Decision Analysis<br/>📚 Source: Saaty (1980)<br/>✅ Why: Holistic evaluation"]
    
    I["🚫 NO Ground Truth"] --> J["⚠️ PROBLEM"]
    J --> K["❌ Can't use Precision/Recall<br/>❌ Can't use F1-score<br/>❌ Can't use mAP<br/>❌ No 'correct answer' available"]
    
    K --> L["💡 SOLUTION: Proxy Metrics"]
    L --> M["🎯 Detection Count Consistency<br/>📊 Confidence Distribution Analysis<br/>⚡ Inference Speed<br/>🔄 Temporal Consistency"]
