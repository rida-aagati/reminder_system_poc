"""
Database Reset Utility for TalentSkout Reminder System
======================================================
Resets db.json to initial pending states for testing reminder triggers and calls.
"""

import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "db.json"

INITIAL_DATA = {
  "interviews": [
    {
      "interview_id": "SCR-101",
      "candidate_name": "Rida Khan",
      "email": "rida.khan@example.com",
      "phone": "+91-9876543210",
      "interview_type": "ai_screener",
      "company": "TalentSkout",
      "role": "Python Backend Developer",
      "duration_minutes": 30,
      "link_sent_at": "2026-10-23T23:30:00+05:30",
      "link_expires_at": "2026-10-25T23:30:00+05:30",
      "time_left_hours": 2.0,
      "assessment_link": "https://screener.talentskout.com/session/scr-101-rida",
      "panelist_name": None,
      "recruiter_name": "Sneha Kapoor",
      "recruiter_email": "sneha.kapoor@talentskout.com",
      "status": "pending",
      "attendance_confirmed": None,
      "email_reminder_sent": True,
      "call_reminder_sent": False
    },
    {
      "interview_id": "SCR-102",
      "candidate_name": "Aarav Sharma",
      "email": "aarav.sharma@example.com",
      "phone": "+91-9811223344",
      "interview_type": "ai_screener",
      "company": "TalentSkout",
      "role": "Data Analyst",
      "duration_minutes": 30,
      "link_sent_at": "2026-10-24T09:30:00+05:30",
      "link_expires_at": "2026-10-26T09:30:00+05:30",
      "time_left_hours": 12.0,
      "assessment_link": "https://screener.talentskout.com/session/scr-102-aarav",
      "panelist_name": None,
      "recruiter_name": "Sneha Kapoor",
      "recruiter_email": "sneha.kapoor@talentskout.com",
      "status": "pending",
      "attendance_confirmed": None,
      "email_reminder_sent": False,
      "call_reminder_sent": False
    },
    {
      "interview_id": "SCR-103",
      "candidate_name": "Pooja Verma",
      "email": "pooja.verma@example.com",
      "phone": "+91-9899001122",
      "interview_type": "ai_screener",
      "company": "TalentSkout",
      "role": "Frontend Engineer",
      "duration_minutes": 30,
      "link_sent_at": "2026-10-24T12:00:00+05:30",
      "link_expires_at": "2026-10-26T12:00:00+05:30",
      "time_left_hours": 14.5,
      "assessment_link": "https://screener.talentskout.com/session/scr-103-pooja",
      "panelist_name": None,
      "recruiter_name": "Rahul Mehta",
      "recruiter_email": "rahul.mehta@talentskout.com",
      "status": "completed",
      "attendance_confirmed": True,
      "email_reminder_sent": True,
      "call_reminder_sent": False
    },
    {
      "interview_id": "PAN-201",
      "candidate_name": "Siddharth Nair",
      "email": "siddharth.nair@example.com",
      "phone": "+91-9822334455",
      "interview_type": "panel_round",
      "company": "TalentSkout",
      "role": "Lead Architect",
      "duration_minutes": 60,
      "scheduled_time": "2026-10-26T01:00:00+05:30",
      "time_left_hours": 3.5,
      "assessment_link": "https://meet.talentskout.com/panel/pan-201-siddharth",
      "panelist_name": "Vikram Malhotra (Engineering VP)",
      "recruiter_name": "Rahul Mehta",
      "recruiter_email": "rahul.mehta@talentskout.com",
      "status": "pending",
      "attendance_confirmed": None,
      "email_reminder_sent": True,
      "call_reminder_sent": False
    },
    {
      "interview_id": "PAN-202",
      "candidate_name": "Meera Joshi",
      "email": "meera.joshi@example.com",
      "phone": "+91-9733445566",
      "interview_type": "panel_round",
      "company": "TalentSkout",
      "role": "Product Designer",
      "duration_minutes": 45,
      "scheduled_time": "2026-10-26T21:30:00+05:30",
      "time_left_hours": 24.0,
      "assessment_link": "https://meet.talentskout.com/panel/pan-202-meera",
      "panelist_name": "Ananya Roy (Design Lead)",
      "recruiter_name": "Sneha Kapoor",
      "recruiter_email": "sneha.kapoor@talentskout.com",
      "status": "pending",
      "attendance_confirmed": None,
      "email_reminder_sent": False,
      "call_reminder_sent": False
    }
  ]
}


def reset_database():
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(INITIAL_DATA, f, indent=2)
    print("✅ [Database Reset] db.json restored to initial pending states.")


if __name__ == "__main__":
    reset_database()
