class LabelCollector:
    """Interactive tool to label scenes"""
    
    def __init__(self, memory_timeline_json: str):
        self.memories = json.load(open(memory_timeline_json))
    
    def label_scene(self, memory_idx: int):
        """Ask human: is this actually important?"""
        memory = self.memories[memory_idx]
        
        # Display image + metadata
        print(f"\nMemory #{memory_idx}")
        print(f"Timestamp: {memory['timestamp']}")
        print(f"Auto-score: {memory['importance']}")
        print(f"Image: {memory['image']}")
        
        # Ask human
        label = input("Is this important? (0=no, 1=maybe, 2=yes, 3=critical): ")
        confidence = input("How confident? (0=unsure, 1=pretty sure, 2=very sure): ")
        comment = input("Why? (optional): ")
        
        return {
            "memory_idx": memory_idx,
            "human_label": int(label),
            "human_confidence": int(confidence),
            "comment": comment,
            "auto_score": memory['importance'],
            "features": {
                "motion": memory.get('motion_mean'),
                "duration": memory.get('duration'),
                "objects": memory.get('objects'),
            }
        }
    
    def collect_labels(self, count: int = 50):
        """Label N random scenes"""
        import random
        indices = random.sample(range(len(self.memories)), count)
        
        labels = []
        for idx in indices:
            label = self.label_scene(idx)
            labels.append(label)
            
            with open("labeled_data.jsonl", "a") as f:
                f.write(json.dumps(label) + "\n")
        
        return labels