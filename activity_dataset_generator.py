import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class ActivityDatasetGenerator:
    """
    Generate realistic human activity datasets with configurable parameters
    to simulate real-world activity patterns.
    """
    
    def __init__(self, config=None):
        """
        Initialize the generator with configuration parameters.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary with the following keys:
            - start_date: Starting date for the dataset (str in 'YYYY-MM-DD' format)
            - end_date: Ending date for the dataset (str in 'YYYY-MM-DD' format)
            - num_participants: Number of unique participants (int)
            - time_resolution: Time resolution in seconds (int)
            - activities: List of activity types to include (list)
            - activity_durations: Dict mapping activities to (min_duration, max_duration) in minutes
            - daily_patterns: Dict mapping activities to hourly probabilities (24-hour array)
            - transition_probs: Dict mapping current activities to next activity probabilities
            - weekend_modifier: Dict modifying activity probabilities on weekends
            - personal_variations: Level of individual variation (float between 0-1)
        """
        # Set default configuration
        self.default_config = {
            'start_date': '2025-01-01',
            'end_date': '2025-01-07',
            'num_participants': 10,
            'time_resolution': 60,  # seconds
            'activities': [
                'walking', 'running', 'sitting', 'standing', 'lying_down', 
                'cycling', 'climbing_stairs', 'descending_stairs', 'jumping',
                'jogging', 'eating', 'typing'
            ],
            'activity_durations': {
                'walking': (5, 60),
                'running': (15, 60),
                'sitting': (15, 180),
                'standing': (2, 30),
                'lying_down': (30, 480),
                'cycling': (15, 120),
                'climbing_stairs': (1, 5),
                'descending_stairs': (1, 5),
                'jumping': (1, 15),
                'jogging': (10, 45),
                'eating': (10, 60),
                'typing': (10, 180)
            },
            'daily_patterns': {
                # Hour-by-hour likelihood for each activity (24 values for 24 hours)
                'walking': [0.02, 0.01, 0.01, 0.01, 0.01, 0.05, 0.10, 0.15, 0.10, 0.05, 0.05, 0.10, 
                           0.15, 0.05, 0.05, 0.05, 0.10, 0.15, 0.10, 0.05, 0.05, 0.03, 0.02, 0.02],
                'running': [0.01, 0.00, 0.00, 0.00, 0.00, 0.15, 0.20, 0.10, 0.05, 0.01, 0.01, 0.01, 
                           0.01, 0.01, 0.01, 0.01, 0.02, 0.20, 0.15, 0.05, 0.02, 0.01, 0.01, 0.00],
                'sitting': [0.30, 0.10, 0.05, 0.05, 0.05, 0.15, 0.20, 0.30, 0.70, 0.80, 0.80, 0.60, 
                           0.70, 0.80, 0.80, 0.70, 0.60, 0.40, 0.50, 0.60, 0.70, 0.60, 0.50, 0.40],
                'standing': [0.05, 0.02, 0.01, 0.01, 0.01, 0.10, 0.15, 0.15, 0.10, 0.10, 0.10, 0.15, 
                            0.15, 0.10, 0.10, 0.10, 0.15, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05],
                'lying_down': [0.80, 0.95, 0.98, 0.98, 0.95, 0.60, 0.20, 0.05, 0.01, 0.01, 0.01, 0.01, 
                              0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.20, 0.30, 0.40, 0.50, 0.70],
                'cycling': [0.01, 0.00, 0.00, 0.00, 0.00, 0.05, 0.10, 0.15, 0.05, 0.01, 0.01, 0.01, 
                           0.01, 0.01, 0.01, 0.01, 0.05, 0.15, 0.10, 0.05, 0.02, 0.01, 0.01, 0.00],
                'climbing_stairs': [0.01, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.08, 0.06, 0.04, 0.04, 0.05, 
                                   0.06, 0.04, 0.04, 0.05, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01],
                'descending_stairs': [0.01, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.08, 0.06, 0.04, 0.04, 0.05, 
                                     0.06, 0.04, 0.04, 0.05, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01],
                'jumping': [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 
                           0.01, 0.01, 0.01, 0.01, 0.05, 0.04, 0.02, 0.01, 0.01, 0.00, 0.00, 0.00],
                'jogging': [0.01, 0.00, 0.00, 0.00, 0.00, 0.10, 0.15, 0.08, 0.02, 0.01, 0.01, 0.01, 
                           0.01, 0.01, 0.01, 0.01, 0.05, 0.15, 0.10, 0.03, 0.01, 0.01, 0.00, 0.00],
                'eating': [0.05, 0.01, 0.00, 0.00, 0.00, 0.05, 0.20, 0.30, 0.10, 0.05, 0.05, 0.40, 
                          0.50, 0.05, 0.05, 0.05, 0.10, 0.40, 0.50, 0.20, 0.10, 0.05, 0.05, 0.03],
                'typing': [0.05, 0.01, 0.00, 0.00, 0.00, 0.01, 0.05, 0.10, 0.50, 0.70, 0.70, 0.50, 
                          0.40, 0.70, 0.70, 0.60, 0.40, 0.20, 0.10, 0.05, 0.10, 0.05, 0.05, 0.03]
            },
            'transition_probs': {
                # Probability of transitioning from one activity to another
                'walking': {
                    'walking': 0.10, 'running': 0.05, 'sitting': 0.40, 'standing': 0.30,
                    'lying_down': 0.05, 'cycling': 0.05, 'climbing_stairs': 0.15, 
                    'descending_stairs': 0.15, 'jumping': 0.01, 'jogging': 0.05, 
                    'eating': 0.10, 'typing': 0.15
                },
                'running': {
                    'walking': 0.50, 'running': 0.20, 'sitting': 0.15, 'standing': 0.10,
                    'lying_down': 0.05, 'cycling': 0.02, 'climbing_stairs': 0.01, 
                    'descending_stairs': 0.01, 'jumping': 0.05, 'jogging': 0.20, 
                    'eating': 0.01, 'typing': 0.01
                },
                'sitting': {
                    'walking': 0.30, 'running': 0.05, 'sitting': 0.30, 'standing': 0.30,
                    'lying_down': 0.10, 'cycling': 0.05, 'climbing_stairs': 0.03, 
                    'descending_stairs': 0.02, 'jumping': 0.01, 'jogging': 0.03, 
                    'eating': 0.25, 'typing': 0.40
                },
                'standing': {
                    'walking': 0.50, 'running': 0.05, 'sitting': 0.30, 'standing': 0.10,
                    'lying_down': 0.05, 'cycling': 0.02, 'climbing_stairs': 0.10, 
                    'descending_stairs': 0.10, 'jumping': 0.02, 'jogging': 0.05, 
                    'eating': 0.10, 'typing': 0.15
                },
                'lying_down': {
                    'walking': 0.20, 'running': 0.05, 'sitting': 0.30, 'standing': 0.40,
                    'lying_down': 0.30, 'cycling': 0.01, 'climbing_stairs': 0.01, 
                    'descending_stairs': 0.01, 'jumping': 0.01, 'jogging': 0.01, 
                    'eating': 0.05, 'typing': 0.05
                },
                'cycling': {
                    'walking': 0.40, 'running': 0.10, 'sitting': 0.30, 'standing': 0.15,
                    'lying_down': 0.05, 'cycling': 0.20, 'climbing_stairs': 0.01, 
                    'descending_stairs': 0.01, 'jumping': 0.01, 'jogging': 0.05, 
                    'eating': 0.10, 'typing': 0.05
                },
                'climbing_stairs': {
                    'walking': 0.40, 'running': 0.05, 'sitting': 0.20, 'standing': 0.20,
                    'lying_down': 0.02, 'cycling': 0.01, 'climbing_stairs': 0.05, 
                    'descending_stairs': 0.30, 'jumping': 0.01, 'jogging': 0.05, 
                    'eating': 0.05, 'typing': 0.10
                },
                'descending_stairs': {
                    'walking': 0.50, 'running': 0.05, 'sitting': 0.20, 'standing': 0.20,
                    'lying_down': 0.02, 'cycling': 0.01, 'climbing_stairs': 0.20, 
                    'descending_stairs': 0.05, 'jumping': 0.01, 'jogging': 0.03, 
                    'eating': 0.05, 'typing': 0.10
                },
                'jumping': {
                    'walking': 0.40, 'running': 0.20, 'sitting': 0.15, 'standing': 0.15,
                    'lying_down': 0.01, 'cycling': 0.01, 'climbing_stairs': 0.01, 
                    'descending_stairs': 0.01, 'jumping': 0.30, 'jogging': 0.20, 
                    'eating': 0.01, 'typing': 0.01
                },
                'jogging': {
                    'walking': 0.50, 'running': 0.20, 'sitting': 0.15, 'standing': 0.10,
                    'lying_down': 0.05, 'cycling': 0.01, 'climbing_stairs': 0.01, 
                    'descending_stairs': 0.01, 'jumping': 0.05, 'jogging': 0.20, 
                    'eating': 0.01, 'typing': 0.01
                },
                'eating': {
                    'walking': 0.30, 'running': 0.05, 'sitting': 0.50, 'standing': 0.20,
                    'lying_down': 0.10, 'cycling': 0.01, 'climbing_stairs': 0.01, 
                    'descending_stairs': 0.01, 'jumping': 0.01, 'jogging': 0.01, 
                    'eating': 0.10, 'typing': 0.20
                },
                'typing': {
                    'walking': 0.20, 'running': 0.01, 'sitting': 0.60, 'standing': 0.15,
                    'lying_down': 0.05, 'cycling': 0.01, 'climbing_stairs': 0.01, 
                    'descending_stairs': 0.01, 'jumping': 0.01, 'jogging': 0.01, 
                    'eating': 0.15, 'typing': 0.30
                }
            },
            'weekend_modifier': {
                # Modify activity probabilities on weekends
                'walking': 1.5,      # 50% more walking on weekends
                'running': 2.0,      # Twice as much running on weekends
                'sitting': 0.7,      # 30% less sitting
                'standing': 0.8,     # 20% less standing
                'lying_down': 1.3,   # 30% more lying down
                'cycling': 3.0,      # Triple cycling
                'climbing_stairs': 0.5,
                'descending_stairs': 0.5,
                'jumping': 1.5,
                'jogging': 2.0,
                'eating': 1.2,
                'typing': 0.4        # 60% less typing (work)
            },
            'personal_variations': 0.3  # 0-1 scale of individual participant variations
        }
        
        # Update with user-provided config if provided
        self.config = self.default_config.copy()
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
    
    def _get_next_activity(self, current_activity, current_time, participant_id):
        """
        Determine the next activity based on transition probabilities,
        time of day, day of week, and individual variations.
        """
        hour = current_time.hour
        is_weekend = current_time.weekday() >= 5  # 5 = Saturday, 6 = Sunday
        
        # Get base transition probabilities for current activity
        transition_probs = self.config['transition_probs'][current_activity].copy()
        
        # Apply time-of-day factors
        for activity in transition_probs:
            time_factor = self.config['daily_patterns'][activity][hour]
            transition_probs[activity] *= time_factor
        
        # Apply weekend modifiers if it's a weekend
        if is_weekend:
            for activity in transition_probs:
                weekend_mod = self.config['weekend_modifier'].get(activity, 1.0)
                transition_probs[activity] *= weekend_mod
        
        # Apply individual variations based on participant_id
        # Use participant_id as a seed for consistent behavior
        np.random.seed(participant_id)
        variation_level = self.config['personal_variations']
        for activity in transition_probs:
            # Individual variation factor between (1-variation_level) and (1+variation_level)
            individual_factor = 1.0 + variation_level * (2 * np.random.random() - 1)
            transition_probs[activity] *= individual_factor
        
        # Reset the random seed
        np.random.seed()
        
        # Normalize probabilities
        total = sum(transition_probs.values())
        normalized_probs = {k: v/total for k, v in transition_probs.items()}
        
        # Select next activity based on probabilities
        activities = list(normalized_probs.keys())
        probabilities = list(normalized_probs.values())
        next_activity = np.random.choice(activities, p=probabilities)
        
        return next_activity
    
    def _get_activity_duration(self, activity, current_time, participant_id):
        """
        Determine the duration of an activity based on typical durations,
        time of day, day of week, and individual variations.
        """
        min_duration, max_duration = self.config['activity_durations'][activity]
        
        # Convert to minutes
        min_minutes, max_minutes = min_duration, max_duration
        
        hour = current_time.hour
        is_weekend = current_time.weekday() >= 5
        
        # Adjust duration based on time of day
        time_factor = self.config['daily_patterns'][activity][hour]
        duration_range = max_minutes - min_minutes
        
        # Base duration (biased toward typical duration for the activity)
        base_duration = min_minutes + duration_range * (0.5 + 0.5 * time_factor)
        
        # Weekend modifier
        if is_weekend:
            weekend_mod = self.config['weekend_modifier'].get(activity, 1.0)
            if weekend_mod > 1:  # Activities people do more on weekends tend to last longer
                base_duration *= min(weekend_mod, 1.5)  # Cap at 50% increase
        
        # Individual variation
        np.random.seed(participant_id)
        variation_level = self.config['personal_variations']
        individual_factor = 1.0 + variation_level * (2 * np.random.random() - 1)
        np.random.seed()
        
        # Final duration in minutes, with some natural randomness
        duration_minutes = max(min_minutes, min(max_minutes, 
                                            int(base_duration * individual_factor * random.uniform(0.8, 1.2))))
        
        # Convert to seconds for the dataset
        return duration_minutes * 60
    
    def generate_dataset(self):
        """
        Generate the complete dataset based on configuration parameters.
        
        Returns:
        --------
        DataFrame with columns: timestamp, participant_id, activity
        """
        start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
        num_participants = self.config['num_participants']
        
        records = []
        
        # Generate data for each participant
        for participant_id in range(1, num_participants + 1):
            # Start with a random activity weighted by time of day
            current_time = start_date
            hour = current_time.hour
            
            # Initial activity probabilities based on time of day
            initial_probs = {act: self.config['daily_patterns'][act][hour] for act in self.config['activities']}
            total = sum(initial_probs.values())
            normalized_probs = {k: v/total for k, v in initial_probs.items()}
            
            activities = list(normalized_probs.keys())
            probabilities = list(normalized_probs.values())
            current_activity = np.random.choice(activities, p=probabilities)
            
            # Generate activities until end date
            while current_time < end_date:
                # Record the current activity
                records.append({
                    'timestamp': current_time,
                    'participant_id': f"P{participant_id:03d}",
                    'activity': current_activity
                })
                
                # Calculate duration of this activity
                duration_seconds = self._get_activity_duration(current_activity, current_time, participant_id)
                
                # Ensure we record at the specified time resolution
                time_resolution = self.config['time_resolution']
                num_records = int(duration_seconds / time_resolution)
                
                for i in range(1, num_records):
                    record_time = current_time + timedelta(seconds=i * time_resolution)
                    if record_time >= end_date:
                        break
                    
                    records.append({
                        'timestamp': record_time,
                        'participant_id': f"P{participant_id:03d}",
                        'activity': current_activity
                    })
                
                # Move to next time slot and activity
                current_time = current_time + timedelta(seconds=duration_seconds)
                if current_time >= end_date:
                    break
                    
                current_activity = self._get_next_activity(current_activity, current_time, participant_id)
        
        # Create DataFrame and sort by timestamp
        df = pd.DataFrame(records)
        df = df.sort_values(['timestamp', 'participant_id']).reset_index(drop=True)
        
        return df

# Example usage
if __name__ == "__main__":
    # Example configuration
    example_config = {
        'start_date': '2025-01-01',
        'end_date': '2025-01-03',
        'num_participants': 5,
        'time_resolution': 300,  # 5 minutes
        'personal_variations': 0.5  # Higher individual variations
    }
    
    # Generate dataset with example config
    generator = ActivityDatasetGenerator(example_config)
    df = generator.generate_dataset()
    
    # Display sample
    print(f"Generated {len(df)} records for {example_config['num_participants']} participants")
    print("\nSample data:")
    print(df.head(10))
    
    # Save to CSV
    df.to_csv('human_activity_dataset.csv', index=False)
    print("\nDataset saved to 'human_activity_dataset.csv'")
    
    # Some basic statistics
    print("\nActivity distribution:")
    activity_counts = df['activity'].value_counts(normalize=True) * 100
    for activity, percentage in activity_counts.items():
        print(f"{activity}: {percentage:.1f}%")
