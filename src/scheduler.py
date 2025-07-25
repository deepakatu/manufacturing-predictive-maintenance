
"""
Maintenance Scheduling System for Predictive Maintenance.
Optimizes maintenance schedules based on equipment health, predictions, and constraints.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaintenanceType(Enum):
    """Types of maintenance activities."""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"

class Priority(Enum):
    """Priority levels for maintenance tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class MaintenanceTask:
    """Represents a maintenance task."""
    task_id: str
    equipment_id: str
    task_type: MaintenanceType
    priority: Priority
    estimated_duration: int  # minutes
    required_skills: List[str]
    required_parts: List[str]
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    status: str = "pending"  # pending, scheduled, in_progress, completed, cancelled
    description: str = ""
    cost_estimate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value > other.priority.value  # Higher priority first

@dataclass
class Technician:
    """Represents a maintenance technician."""
    technician_id: str
    name: str
    skills: List[str]
    availability_start: datetime
    availability_end: datetime
    hourly_rate: float
    current_tasks: List[str] = field(default_factory=list)
    
    def is_available(self, start_time: datetime, duration_minutes: int) -> bool:
        """Check if technician is available for a task."""
        end_time = start_time + timedelta(minutes=duration_minutes)
        return (start_time >= self.availability_start and 
                end_time <= self.availability_end and
                len(self.current_tasks) == 0)  # Simplified: one task at a time
    
    def has_skills(self, required_skills: List[str]) -> bool:
        """Check if technician has required skills."""
        return all(skill in self.skills for skill in required_skills)

@dataclass
class Equipment:
    """Represents equipment for maintenance scheduling."""
    equipment_id: str
    name: str
    location: str
    equipment_type: str
    criticality: Priority
    current_health: float
    predicted_rul: Optional[float] = None
    last_maintenance: Optional[datetime] = None
    maintenance_interval: int = 2160  # hours (3 months)
    
    def needs_maintenance(self) -> bool:
        """Check if equipment needs maintenance based on schedule."""
        if not self.last_maintenance:
            return True
        
        time_since_maintenance = datetime.now() - self.last_maintenance
        return time_since_maintenance.total_seconds() / 3600 > self.maintenance_interval

class MaintenanceScheduler:
    """
    Intelligent maintenance scheduler that optimizes maintenance activities
    based on equipment health, predictions, resource availability, and constraints.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.tasks = []
        self.technicians = []
        self.equipment = {}
        self.schedule = {}  # date -> list of tasks
        self.task_queue = []  # Priority queue for urgent tasks
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load scheduler configuration."""
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                logger.warning(f"Config file {config_file} not found, using defaults")
        
        return {
            'working_hours': {'start': 8, 'end': 17},
            'working_days': [0, 1, 2, 3, 4],  # Monday to Friday
            'emergency_threshold': 0.2,  # Health score below which emergency maintenance is triggered
            'predictive_threshold': 0.5,  # Health score below which predictive maintenance is scheduled
            'planning_horizon_days': 30,
            'optimization_weights': {
                'cost': 0.3,
                'urgency': 0.4,
                'resource_utilization': 0.3
            }
        }
    
    def add_equipment(self, equipment: Equipment):
        """Add equipment to the scheduler."""
        self.equipment[equipment.equipment_id] = equipment
        logger.info(f"Added equipment: {equipment.equipment_id}")
    
    def add_technician(self, technician: Technician):
        """Add technician to the scheduler."""
        self.technicians.append(technician)
        logger.info(f"Added technician: {technician.name}")
    
    def create_task_from_prediction(self, equipment_id: str, health_score: float, 
                                  predicted_failure: str, rul_hours: Optional[float] = None) -> MaintenanceTask:
        """Create maintenance task based on predictive analytics."""
        equipment = self.equipment.get(equipment_id)
        if not equipment:
            raise ValueError(f"Equipment {equipment_id} not found")
        
        # Determine task type and priority based on health score and predictions
        if health_score < self.config['emergency_threshold']:
            task_type = MaintenanceType.EMERGENCY
            priority = Priority.CRITICAL
            description = f"Emergency maintenance required - Health: {health_score:.2f}"
        elif health_score < self.config['predictive_threshold']:
            task_type = MaintenanceType.PREDICTIVE
            priority = Priority.HIGH if health_score < 0.3 else Priority.MEDIUM
            description = f"Predictive maintenance - {predicted_failure} predicted"
        else:
            task_type = MaintenanceType.PREVENTIVE
            priority = Priority.LOW
            description = f"Scheduled preventive maintenance"
        
        # Estimate duration based on failure type and equipment
        duration_map = {
            'bearing_wear': 240,  # 4 hours
            'misalignment': 120,  # 2 hours
            'overheating': 180,   # 3 hours
            'normal': 60          # 1 hour
        }
        estimated_duration = duration_map.get(predicted_failure, 120)
        
        # Determine required skills
        skill_map = {
            'bearing_wear': ['mechanical', 'vibration_analysis'],
            'misalignment': ['mechanical', 'alignment'],
            'overheating': ['electrical', 'thermal_analysis'],
            'normal': ['general_maintenance']
        }
        required_skills = skill_map.get(predicted_failure, ['general_maintenance'])
        
        # Create task
        task = MaintenanceTask(
            task_id=f"TASK_{equipment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            equipment_id=equipment_id,
            task_type=task_type,
            priority=priority,
            estimated_duration=estimated_duration,
            required_skills=required_skills,
            required_parts=[],  # Would be determined by failure analysis
            description=description,
            cost_estimate=self._estimate_cost(estimated_duration, required_skills)
        )
        
        return task
    
    def _estimate_cost(self, duration_minutes: int, required_skills: List[str]) -> float:
        """Estimate cost of maintenance task."""
        # Base cost calculation
        base_hourly_rate = 75.0  # Average technician rate
        labor_cost = (duration_minutes / 60) * base_hourly_rate
        
        # Skill premium
        skill_premium = {
            'vibration_analysis': 1.2,
            'thermal_analysis': 1.15,
            'electrical': 1.1,
            'mechanical': 1.0,
            'general_maintenance': 0.9
        }
        
        max_premium = max([skill_premium.get(skill, 1.0) for skill in required_skills])
        labor_cost *= max_premium
        
        # Add estimated parts cost (simplified)
        parts_cost = labor_cost * 0.3  # Assume parts cost is 30% of labor
        
        return labor_cost + parts_cost
    
    def add_task(self, task: MaintenanceTask):
        """Add task to the scheduler."""
        self.tasks.append(task)
        
        # Add to priority queue if urgent
        if task.priority in [Priority.CRITICAL, Priority.HIGH]:
            heapq.heappush(self.task_queue, task)
        
        logger.info(f"Added task: {task.task_id} (Priority: {task.priority.name})")
    
    def find_available_technician(self, task: MaintenanceTask, 
                                 preferred_start: datetime) -> Optional[Tuple[Technician, datetime]]:
        """Find available technician for a task."""
        suitable_technicians = [
            tech for tech in self.technicians 
            if tech.has_skills(task.required_skills)
        ]
        
        if not suitable_technicians:
            return None
        
        # Sort by availability and cost
        suitable_technicians.sort(key=lambda t: t.hourly_rate)
        
        # Find earliest available slot
        for technician in suitable_technicians:
            # Check availability at preferred time
            if technician.is_available(preferred_start, task.estimated_duration):
                return technician, preferred_start
            
            # Find next available slot within working hours
            current_time = preferred_start
            end_time = preferred_start + timedelta(days=7)  # Look ahead 1 week
            
            while current_time < end_time:
                if self._is_working_time(current_time):
                    if technician.is_available(current_time, task.estimated_duration):
                        return technician, current_time
                
                current_time += timedelta(hours=1)
        
        return None
    
    def _is_working_time(self, dt: datetime) -> bool:
        """Check if datetime falls within working hours."""
        if dt.weekday() not in self.config['working_days']:
            return False
        
        working_start = self.config['working_hours']['start']
        working_end = self.config['working_hours']['end']
        
        return working_start <= dt.hour < working_end
    
    def schedule_task(self, task: MaintenanceTask, 
                     preferred_start: Optional[datetime] = None) -> bool:
        """Schedule a maintenance task."""
        if preferred_start is None:
            if task.priority == Priority.CRITICAL:
                preferred_start = datetime.now()
            else:
                preferred_start = datetime.now() + timedelta(hours=24)
        
        # Find available technician
        assignment = self.find_available_technician(task, preferred_start)
        
        if not assignment:
            logger.warning(f"No available technician found for task {task.task_id}")
            return False
        
        technician, start_time = assignment
        end_time = start_time + timedelta(minutes=task.estimated_duration)
        
        # Schedule the task
        task.scheduled_start = start_time
        task.scheduled_end = end_time
        task.status = "scheduled"
        
        # Assign to technician
        technician.current_tasks.append(task.task_id)
        
        # Add to schedule
        schedule_date = start_time.date()
        if schedule_date not in self.schedule:
            self.schedule[schedule_date] = []
        self.schedule[schedule_date].append(task)
        
        logger.info(f"Scheduled task {task.task_id} for {start_time} with {technician.name}")
        return True
    
    def optimize_schedule(self, days_ahead: int = 7) -> Dict[str, Any]:
        """Optimize maintenance schedule using heuristic algorithms."""
        logger.info(f"Optimizing schedule for next {days_ahead} days")
        
        # Get unscheduled tasks
        unscheduled_tasks = [task for task in self.tasks if task.status == "pending"]
        
        # Sort tasks by priority and urgency
        unscheduled_tasks.sort(key=lambda t: (
            -t.priority.value,  # Higher priority first
            t.created_at        # Earlier created first
        ))
        
        scheduled_count = 0
        failed_count = 0
        
        for task in unscheduled_tasks:
            # Determine preferred start time based on urgency
            if task.priority == Priority.CRITICAL:
                preferred_start = datetime.now()
            elif task.priority == Priority.HIGH:
                preferred_start = datetime.now() + timedelta(hours=4)
            else:
                preferred_start = datetime.now() + timedelta(days=1)
            
            if self.schedule_task(task, preferred_start):
                scheduled_count += 1
            else:
                failed_count += 1
        
        # Calculate optimization metrics
        total_cost = sum(task.cost_estimate for task in self.tasks if task.status == "scheduled")
        
        # Resource utilization
        total_technician_hours = sum(
            len(tech.current_tasks) * 8 for tech in self.technicians  # Assume 8 hours per task
        )
        available_technician_hours = len(self.technicians) * 8 * days_ahead
        utilization = total_technician_hours / available_technician_hours if available_technician_hours > 0 else 0
        
        optimization_result = {
            'scheduled_tasks': scheduled_count,
            'failed_to_schedule': failed_count,
            'total_estimated_cost': total_cost,
            'resource_utilization': utilization,
            'schedule_efficiency': scheduled_count / len(unscheduled_tasks) if unscheduled_tasks else 1.0
        }
        
        logger.info(f"Optimization completed: {optimization_result}")
        return optimization_result
    
    def get_schedule_for_date(self, date: datetime) -> List[MaintenanceTask]:
        """Get scheduled tasks for a specific date."""
        return self.schedule.get(date.date(), [])
    
    def get_technician_schedule(self, technician_id: str, days: int = 7) -> List[MaintenanceTask]:
        """Get schedule for a specific technician."""
        technician_tasks = []
        
        start_date = datetime.now().date()
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            daily_tasks = self.schedule.get(current_date, [])
            
            for task in daily_tasks:
                # Find assigned technician
                for tech in self.technicians:
                    if task.task_id in tech.current_tasks and tech.technician_id == technician_id:
                        technician_tasks.append(task)
                        break
        
        return technician_tasks
    
    def update_equipment_health(self, equipment_id: str, health_score: float, 
                               predicted_failure: str = "normal", rul_hours: Optional[float] = None):
        """Update equipment health and create maintenance tasks if needed."""
        if equipment_id not in self.equipment:
            logger.warning(f"Equipment {equipment_id} not found")
            return
        
        equipment = self.equipment[equipment_id]
        equipment.current_health = health_score
        equipment.predicted_rul = rul_hours
        
        # Check if maintenance task should be created
        should_create_task = False
        
        if health_score < self.config['emergency_threshold']:
            should_create_task = True
        elif health_score < self.config['predictive_threshold']:
            # Check if there's already a pending task for this equipment
            existing_tasks = [
                task for task in self.tasks 
                if task.equipment_id == equipment_id and task.status in ["pending", "scheduled"]
            ]
            if not existing_tasks:
                should_create_task = True
        
        if should_create_task:
            task = self.create_task_from_prediction(
                equipment_id, health_score, predicted_failure, rul_hours
            )
            self.add_task(task)
            
            # Auto-schedule critical tasks
            if task.priority == Priority.CRITICAL:
                self.schedule_task(task)
    
    def generate_maintenance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive maintenance report."""
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=days)
        
        # Collect tasks in date range
        report_tasks = []
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            daily_tasks = self.schedule.get(current_date, [])
            report_tasks.extend(daily_tasks)
        
        # Calculate metrics
        total_tasks = len(report_tasks)
        tasks_by_type = defaultdict(int)
        tasks_by_priority = defaultdict(int)
        total_cost = 0
        total_duration = 0
        
        for task in report_tasks:
            tasks_by_type[task.task_type.value] += 1
            tasks_by_priority[task.priority.name] += 1
            total_cost += task.cost_estimate
            total_duration += task.estimated_duration
        
        # Equipment health summary
        equipment_summary = []
        for eq_id, equipment in self.equipment.items():
            equipment_summary.append({
                'equipment_id': eq_id,
                'name': equipment.name,
                'health_score': equipment.current_health,
                'predicted_rul': equipment.predicted_rul,
                'criticality': equipment.criticality.name,
                'needs_maintenance': equipment.needs_maintenance()
            })
        
        # Technician utilization
        technician_utilization = []
        for tech in self.technicians:
            assigned_tasks = len(tech.current_tasks)
            utilization = min(100, (assigned_tasks / (days / 7)) * 100)  # Rough calculation
            technician_utilization.append({
                'technician_id': tech.technician_id,
                'name': tech.name,
                'skills': tech.skills,
                'assigned_tasks': assigned_tasks,
                'utilization_percent': utilization
            })
        
        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': days
            },
            'task_summary': {
                'total_tasks': total_tasks,
                'tasks_by_type': dict(tasks_by_type),
                'tasks_by_priority': dict(tasks_by_priority),
                'total_estimated_cost': total_cost,
                'total_estimated_duration_hours': total_duration / 60
            },
            'equipment_summary': equipment_summary,
            'technician_utilization': technician_utilization,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate maintenance recommendations based on current state."""
        recommendations = []
        
        # Check for overdue maintenance
        overdue_equipment = [
            eq for eq in self.equipment.values() 
            if eq.needs_maintenance() and eq.current_health < 0.7
        ]
        
        if overdue_equipment:
            recommendations.append(
                f"Schedule maintenance for {len(overdue_equipment)} equipment units with overdue maintenance"
            )
        
        # Check for critical health scores
        critical_equipment = [
            eq for eq in self.equipment.values() 
            if eq.current_health < self.config['emergency_threshold']
        ]
        
        if critical_equipment:
            recommendations.append(
                f"Immediate attention required for {len(critical_equipment)} equipment units with critical health scores"
            )
        
        # Check technician workload
        overloaded_technicians = [
            tech for tech in self.technicians 
            if len(tech.current_tasks) > 5
        ]
        
        if overloaded_technicians:
            recommendations.append(
                f"Consider redistributing workload for {len(overloaded_technicians)} overloaded technicians"
            )
        
        # Check for skill gaps
        pending_tasks = [task for task in self.tasks if task.status == "pending"]
        required_skills = set()
        for task in pending_tasks:
            required_skills.update(task.required_skills)
        
        available_skills = set()
        for tech in self.technicians:
            available_skills.update(tech.skills)
        
        missing_skills = required_skills - available_skills
        if missing_skills:
            recommendations.append(
                f"Consider training or hiring for missing skills: {', '.join(missing_skills)}"
            )
        
        return recommendations

def main():
    """Example usage of the maintenance scheduler."""
    # Create scheduler
    scheduler = MaintenanceScheduler()
    
    # Add equipment
    equipment1 = Equipment(
        equipment_id="PUMP_001",
        name="Main Circulation Pump",
        location="Production Floor A",
        equipment_type="pump",
        criticality=Priority.HIGH,
        current_health=0.75,
        predicted_rul=500.0
    )
    
    equipment2 = Equipment(
        equipment_id="MOTOR_002",
        name="Drive Motor",
        location="Production Floor B",
        equipment_type="motor",
        criticality=Priority.MEDIUM,
        current_health=0.45,  # Low health
        predicted_rul=200.0
    )
    
    scheduler.add_equipment(equipment1)
    scheduler.add_equipment(equipment2)
    
    # Add technicians
    tech1 = Technician(
        technician_id="TECH_001",
        name="John Smith",
        skills=["mechanical", "vibration_analysis", "general_maintenance"],
        availability_start=datetime.now(),
        availability_end=datetime.now() + timedelta(days=30),
        hourly_rate=75.0
    )
    
    tech2 = Technician(
        technician_id="TECH_002",
        name="Jane Doe",
        skills=["electrical", "thermal_analysis", "general_maintenance"],
        availability_start=datetime.now(),
        availability_end=datetime.now() + timedelta(days=30),
        hourly_rate=80.0
    )
    
    scheduler.add_technician(tech1)
    scheduler.add_technician(tech2)
    
    # Update equipment health (simulating predictions)
    scheduler.update_equipment_health("PUMP_001", 0.75, "normal")
    scheduler.update_equipment_health("MOTOR_002", 0.45, "bearing_wear", 200.0)
    
    # Optimize schedule
    optimization_result = scheduler.optimize_schedule(days_ahead=7)
    print("Optimization Result:", optimization_result)
    
    # Generate report
    report = scheduler.generate_maintenance_report(days=30)
    print("\nMaintenance Report:")
    print(f"Total tasks: {report['task_summary']['total_tasks']}")
    print(f"Total cost: ${report['task_summary']['total_estimated_cost']:.2f}")
    print(f"Equipment needing attention: {len([eq for eq in report['equipment_summary'] if eq['health_score'] < 0.5])}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    main()
