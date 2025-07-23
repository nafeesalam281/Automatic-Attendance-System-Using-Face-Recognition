# Run this separately to generate a weekly summary
import pandas as pd

df = pd.read_csv("monthly_report.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Filter last 7 days
last_7_days = df[df["Date"] >= pd.Timestamp.now() - pd.Timedelta(days=7)]

# Calculate attendance %
summary = last_7_days.groupby("Name")["Status"].value_counts().unstack(fill_value=0)
summary["Total_Days"] = summary.sum(axis=1)
summary["Attendance (%)"] = (summary.get("Present", 0) / summary["Total_Days"] * 100).round(2)
summary = summary.reset_index()

summary.to_csv("weekly_attendance_report.csv", index=False)
print("📁 Weekly report saved → weekly_attendance_report.csv")
