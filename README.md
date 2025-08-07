## 🔍 Why Use These Methods Without Ground Truth?

```mermaid
graph TD
    A["❓ WHY are these methods needed?"] --> B["🎯 PROBLEM: No Ground Truth"]
    
    B --> C["🔬 SCIENTIFIC FOUNDATIONS"]
    
    C --> D["📊 Cross-Validation<br/>🎓 Theory: Machine Learning<br/>📚 Hastie et al. (2009)<br/>✅ Estimates generalization error"]
    
    C --> E["🧮 Statistical Significance<br/>🎓 Hypothesis Testing<br/>📚 Student's t-test (1908)<br/>✅ Reduces randomness"]
    
    C --> F["🛡️ Robustness Test<br/>🎓 Robust Statistics<br/>📚 Huber (1981)<br/>✅ Tests stability under perturbations"]
    
    C --> G["📏 Confidence Intervals<br/>🎓 Statistical Inference<br/>📚 Neyman (1937)<br/>✅ Quantifies uncertainty"]
    
    C --> H["🎯 Combined Scoring<br/>🎓 Multi-criteria Decision Analysis<br/>📚 Saaty (1980)<br/>✅ Enables holistic evaluation"]
    
    I["🚫 NO Ground Truth"] --> J["⚠️ KEY PROBLEM"]
    
    J --> K["❌ Can't use Precision/Recall<br/>❌ Can't use F1-score<br/>❌ Can't use mAP<br/>❌ No 'correct answer' available"]
    
    K --> L["💡 SOLUTION: Proxy Metrics"]
    
    L --> M["🎯 Detection Count Consistency<br/>📊 Confidence Distribution<br/>⚡ Inference Speed<br/>🔄 Temporal Consistency"]
