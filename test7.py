import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


def main():
    st.set_page_config(page_title="Lead Generation Analysis", layout="wide")
    st.title("Lead Generation Data Analysis Dashboard")
    st.markdown(
        "### Interactive analysis of lead generation performance across associates"
    )

    # Use a specific file path instead of asking for upload
    file_path = "Data Assignment.xlsx"

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Efficiency Analysis",
            "Performance Variability",
            "Time Management",
            "Attendance Impact",
            "Incomplete Leads Trend",
            "Performance Consistency",
            "High-Performance Days",
            "Advanced Analysis",
        ]
    )

    try:
        # Load the Excel file
        xls = pd.ExcelFile(file_path)

        # Store all dataframes in a dictionary
        sheets = {}
        for sheet in xls.sheet_names:
            sheets[sheet] = pd.read_excel(file_path, sheet_name=sheet)

        # ------ TAB 1: EFFICIENCY ANALYSIS ------
        with tab1:
            st.header("Lead Generation Efficiency Analysis")

            # Dictionary to hold efficiencies
            efficiencies = {}

            # Process each associate
            for sheet in xls.sheet_names:
                df = sheets[sheet]

                # Data Cleaning
                df = df.dropna(subset=["Leads", "Time spent on LG (mins)"])
                df["Leads"] = pd.to_numeric(df["Leads"], errors="coerce").fillna(0)
                df["Time spent on LG (mins)"] = pd.to_numeric(df["Time spent on LG (mins)"], errors="coerce").fillna(0)

                # Calculate total leads and total time
                total_leads = df["Leads"].sum()
                total_time = df["Time spent on LG (mins)"].sum()

                # Avoid division by zero
                if total_time > 0:
                    efficiency = total_leads / total_time
                else:
                    efficiency = 0

                efficiencies[sheet] = efficiency

            # Find the associate with the highest efficiency
            most_efficient = max(efficiencies, key=efficiencies.get)

            # Display the efficiencies in a dataframe
            efficiency_df = pd.DataFrame({
                'Associate': list(efficiencies.keys()),
                'Efficiency (Leads/Minute)': list(efficiencies.values())
            })

            st.dataframe(efficiency_df)

            st.subheader("Lead Generation Efficiency Visualization")

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(efficiencies.keys(), efficiencies.values(), color="skyblue")
            plt.axhline(y=efficiencies[most_efficient], color='red', linestyle='--', label=f'Highest: {most_efficient}')
            plt.title("Lead Generation Efficiency per Associate")
            plt.xlabel("Associate")
            plt.ylabel("Efficiency (Leads per Minute)")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.legend()
            st.pyplot(fig)

            st.info(f"🏆 Most Efficient Associate: **{most_efficient}** with {efficiencies[most_efficient]:.4f} leads per minute")
            st.markdown("""
            **Inference:**
            - The bar chart shows the efficiency of each associate in generating leads, measured as leads per minute.
            - The red dashed line indicates the highest efficiency level achieved.
            - Higher efficiency associates are able to generate more leads in less time, potentially indicating better skills or techniques.
            - Associates with lower efficiency might need additional training or process optimization.
            """)

        # ------ TAB 2: PERFORMANCE VARIABILITY ------
        with tab2:
            st.header("Daily Performance Variability Analysis")

            # Dictionary to hold standard deviation values
            variability = {}

            # Process each associate again
            for sheet in xls.sheet_names:
                df = sheets[sheet]

                # Data Cleaning
                df = df.dropna(subset=["Leads"])
                df["Leads"] = pd.to_numeric(df["Leads"], errors="coerce").fillna(0)

                # Calculate standard deviation of leads
                std_dev_leads = df["Leads"].std()
                variability[sheet] = std_dev_leads

            # Find the associate with the highest variability
            most_variable = max(variability, key=variability.get)

            # Display the variability in a dataframe
            variability_df = pd.DataFrame({
                'Associate': list(variability.keys()),
                'Standard Deviation of Leads': list(variability.values())
            })

            st.dataframe(variability_df)

            st.subheader("Performance Variability Visualization")

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(variability.keys(), variability.values(), color="salmon")
            plt.axhline(y=variability[most_variable], color='red', linestyle='--', label=f'Highest Variability: {most_variable}')
            plt.title("Daily Performance Variability per Associate")
            plt.xlabel("Associate")
            plt.ylabel("Standard Deviation of Leads")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.legend()
            st.pyplot(fig)

            st.info(f"🎯 Associate with Highest Variability: **{most_variable}** with std dev of {variability[most_variable]:.4f}")
            st.markdown("""
            **Inference:**
            - This chart shows the consistency of each associate's performance measured by standard deviation of daily leads.
            - Higher standard deviation indicates more variable (less consistent) performance from day to day.
            - Associates with high variability may need additional support to stabilize their performance.
            - More consistent performers (lower bars) have more predictable output which can be beneficial for resource planning.
            """)

        # ------ TAB 3: TIME MANAGEMENT ANALYSIS ------
        with tab3:
            st.header("Time Management Analysis")

            # Dictionary to hold correlation coefficients
            correlations = {}

            # Process each associate again
            for sheet in xls.sheet_names:
                df = sheets[sheet]

                # Check if columns exist properly
                if "Avg Time Per Lead (mins)" in df.columns and "Leads" in df.columns:
                    df["Avg Time Per Lead (mins)"] = pd.to_numeric(df["Avg Time Per Lead (mins)"], errors="coerce")
                    df["Leads"] = pd.to_numeric(df["Leads"], errors="coerce")

                    df = df.dropna(subset=["Avg Time Per Lead (mins)", "Leads"])

                    if not df.empty:
                        correlation = df["Avg Time Per Lead (mins)"].corr(df["Leads"])
                    else:
                        correlation = 0
                else:
                    correlation = 0

                correlations[sheet] = correlation

            # Display the correlations in a dataframe
            correlation_df = pd.DataFrame({
                'Associate': list(correlations.keys()),
                'Correlation Coefficient': list(correlations.values())
            })

            st.dataframe(correlation_df)

            st.subheader("Time Management Correlation")

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(correlations.keys(), correlations.values(), color="mediumseagreen")
            plt.axhline(0, color="black", linestyle="-")
            plt.title("Time Management Correlation per Associate")
            plt.xlabel("Associate")
            plt.ylabel("Correlation Coefficient")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            st.pyplot(fig)

            st.markdown("""
            **Interpretation Guide:**
            - Correlation close to -1: Strong negative relation (more time per lead → fewer total leads)
            - Correlation close to 0: No strong relation
            - Correlation close to +1: Strong positive relation (more time per lead → more total leads)
            
            **Inference:**
            - This chart shows how the average time spent per lead correlates with the total number of leads generated.
            - Negative correlations suggest that spending less time per lead results in more leads overall (efficiency).
            - Positive correlations might indicate that spending more time on each lead yields better results.
            - Associates with correlations near zero may not have a consistent relationship between time spent and results.
            """)

            # Impact of Longer Lead Generation Time
            st.header("Impact of Longer Lead Generation Time")

            associate_selector = st.selectbox(
                "Select Associate for Optimal Time Analysis",
                options=xls.sheet_names
            )

            df = sheets[associate_selector]

            if "Time spent on LG (mins)" in df.columns and "Leads" in df.columns:
                df["Time spent on LG (mins)"] = pd.to_numeric(df["Time spent on LG (mins)"], errors="coerce").fillna(0)
                df["Leads"] = pd.to_numeric(df["Leads"], errors="coerce").fillna(0)

                df = df[(df["Time spent on LG (mins)"] > 0) & (df["Leads"] >= 0)]

                if not df.empty:
                    # Binning the 'Time spent on LG' to find patterns
                    df["Time Bin"] = pd.cut(
                        df["Time spent on LG (mins)"],
                        bins=np.arange(0, df["Time spent on LG (mins)"].max() + 30, 30)
                    )

                    # Average leads per bin
                    bin_stats = df.groupby("Time Bin")["Leads"].mean()

                    # Find the bin with highest average leads
                    optimal_bin = bin_stats.idxmax()
                    optimal_time_midpoint = (optimal_bin.left + optimal_bin.right) / 2

                    fig, ax = plt.subplots(figsize=(12, 6))
                    bin_stats.plot(kind="bar", color="mediumturquoise", edgecolor="black", ax=ax)
                    plt.axhline(bin_stats.max(), color="red", linestyle="--", label="Peak Avg Leads Bin")
                    plt.title(f"Lead Generation vs Time Spent - {associate_selector}")
                    plt.xlabel("Time Spent on LG (mins) Bins")
                    plt.ylabel("Average Leads")
                    plt.xticks(rotation=45)
                    plt.grid(axis="y", linestyle="--", alpha=0.7)
                    plt.legend()
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.success(f"Optimal Time for {associate_selector}: ~{optimal_time_midpoint:.2f} minutes")
                    st.markdown(f"""
                    **Inference:**
                    - The chart shows how the average number of leads varies with different time spent on lead generation.
                    - The bin with the highest average leads (indicated by the red line) represents the optimal time range.
                    - For {associate_selector}, spending around {optimal_time_midpoint:.2f} minutes on lead generation tends to yield the best results.
                    - Time spent significantly above or below this optimal range may result in diminishing returns.
                    """)
                else:
                    st.warning(f"Insufficient valid data for analysis for {associate_selector}.")
            else:
                st.warning(f"Required columns missing for {associate_selector}.")

        # ------ TAB 4: ATTENDANCE IMPACT ANALYSIS ------
        with tab4:
            st.header("Team Review Attendance Impact Analysis")

            # Merge all sheets into a single dataframe
            df_list = []
            for name, data in sheets.items():
                data["Associate"] = name
                df_list.append(data)
            df = pd.concat(df_list, ignore_index=True)

            # Ensure date format
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Check for attendance column
            if "Daily Team Review Attended" in df.columns:
                # Clean attendance data
                df["Daily Team Review Attended"] = df["Daily Team Review Attended"].fillna("Not Attended")
                df["Daily Team Review Attended"] = df["Daily Team Review Attended"].astype(str).str.strip().str.lower()
                df["Team_Review_Attended"] = df["Daily Team Review Attended"].apply(
                    lambda x: "Attended" if "attend" in x.lower() else "Not Attended"
                )
            else:
                # Randomly simulate attendance if not already present
                st.warning("No attendance data found. Using randomly simulated attendance data for demonstration.")
                np.random.seed(42)
                df["Team_Review_Attended"] = np.random.choice(["Attended", "Not Attended"], size=len(df), p=[0.8, 0.2])

            # Compare average leads based on attendance
            attendance_group = df.groupby(["Associate", "Team_Review_Attended"])["Leads"].mean().reset_index()

            # Pivot for easier comparison
            pivot_attendance = attendance_group.pivot(
                index="Associate", columns="Team_Review_Attended", values="Leads"
            )

            # Display the dataframe
            st.dataframe(pivot_attendance)

            # Calculate percentage difference if both columns exist
            if set(["Attended", "Not Attended"]).issubset(pivot_attendance.columns):
                pivot_attendance["% Difference"] = ((pivot_attendance["Attended"] - pivot_attendance["Not Attended"]) / 
                                                 pivot_attendance["Not Attended"] * 100)

                # Bar plot for visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                pivot_attendance[["Attended", "Not Attended"]].plot(
                    kind="bar", color=["lightgreen", "lightcoral"], ax=ax
                )
                plt.title("Average Leads: Attended vs Missed Team Reviews")
                plt.ylabel("Average Leads")
                plt.xlabel("Associate")
                plt.xticks(rotation=45)
                plt.grid(axis="y", alpha=0.3)
                plt.legend(title="Review Attendance")
                plt.tight_layout()
                st.pyplot(fig)

                # Display inferences
                st.subheader("Impact Analysis of Team Review Attendance")

                for associate in pivot_attendance.index:
                    if "% Difference" in pivot_attendance.columns:
                        percentage_diff = pivot_attendance.loc[associate, "% Difference"]

                        if percentage_diff > 0:
                            st.success(f"✅ **{associate}**: Shows an improvement of {percentage_diff:.2f}% in leads when attending team reviews.")
                        elif percentage_diff < 0:
                            st.error(f"⚠️ **{associate}**: Shows a decrease of {abs(percentage_diff):.2f}% in leads when attending team reviews.")
                        else:
                            st.info(f"ℹ️ **{associate}**: Shows no significant difference whether attending or missing team reviews.")

                st.markdown("""
                **Overall Inference:**
                - The chart compares each associate's average lead performance when they attend vs. miss team reviews.
                - Green bars represent performance when attending reviews, red bars when missing reviews.
                - Positive percentage differences suggest that team reviews are beneficial for those associates.
                - Negative differences might indicate that some associates perform better with more independent work time.
                - Management can use this data to optimize team review participation for maximum productivity.
                """)
            else:
                st.warning("Insufficient data to calculate attendance impact. Need both attendance and non-attendance data points.")

        # ------ TAB 5: INCOMPLETE LEADS TREND ------
        with tab5:
            st.header("Incomplete Leads Trend Analysis")

            associate_selector_trend = st.selectbox(
                "Select Associate for Trend Analysis",
                options=xls.sheet_names,
                key="trend_select"
            )

            associate_df = sheets[associate_selector_trend]

            # Sort the data by Date to analyze the trend over time
            if "Date" in associate_df.columns:
                associate_df["Date"] = pd.to_datetime(associate_df["Date"], errors="coerce")
                associate_df = associate_df.sort_values("Date")

                # Drop rows with NaN values in the target variable 'No. of Incomplete Leads'
                if "No. of Incomplete Leads" in associate_df.columns:
                    associate_df = associate_df.dropna(subset=["No. of Incomplete Leads"])
                    associate_df["No. of Incomplete Leads"] = pd.to_numeric(associate_df["No. of Incomplete Leads"], errors="coerce")

                    # Ensure there are enough data points for regression
                    if len(associate_df) > 1:
                        # Prepare regression model
                        X = np.arange(len(associate_df)).reshape(-1, 1)  # Time as increasing numbers
                        y = associate_df["No. of Incomplete Leads"].values  # Number of incomplete leads

                        # Fit the regression model
                        model = LinearRegression()
                        model.fit(X, y)

                        # The slope of the regression line indicates the direction of the trend
                        slope = model.coef_[0]

                        # Classify the trend based on the slope
                        if slope < 0:
                            trend = "Improvement"
                            inference = "The associate is handling incomplete leads better over time, with a decreasing number of incomplete leads."
                        else:
                            trend = "Deterioration"
                            inference = "The associate's performance with incomplete leads is worsening over time."

                        # Line plot with regression
                        fig, ax = plt.subplots(figsize=(12, 6))
                        plt.scatter(associate_df["Date"], y, label="Incomplete Leads", color="blue")
                        plt.plot(associate_df["Date"], model.predict(X), color="red", linewidth=2, label="Trend Line")
                        plt.title(f"Incomplete Leads Over Time: {associate_selector_trend}")
                        plt.xlabel("Date")
                        plt.ylabel("No. of Incomplete Leads")
                        plt.xticks(rotation=45)
                        plt.legend()
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)

                        if slope < 0:
                            st.success(f"**Trend: {trend}** (Slope = {slope:.3f})")
                        else:
                            st.error(f"**Trend: {trend}** (Slope = {slope:.3f})")

                        st.markdown(f"""
                        **Inference:**
                        {inference}
                        
                        - A downward trend (negative slope) indicates improvement in handling leads to completion.
                        - An upward trend (positive slope) suggests more leads are being left incomplete over time.
                        - The steepness of the slope indicates how quickly the performance is changing.
                        """)
                    else:
                        st.warning(f"Not enough data points for {associate_selector_trend} to perform trend analysis.")
                else:
                    st.warning("No 'No. of Incomplete Leads' column found in the data.")
            else:
                st.warning("No 'Date' column found in the data.")

        # ------ TAB 6: PERFORMANCE CONSISTENCY ------
        with tab6:
            st.header("Performance Consistency Analysis")

            # Dictionary to hold CV values
            performance_consistency = {}

            # Process each associate
            for sheet in xls.sheet_names:
                df = sheets[sheet]

                if "Leads" in df.columns:
                    df["Leads"] = pd.to_numeric(df["Leads"], errors="coerce").fillna(0)

                    mean_leads = df["Leads"].mean()
                    std_leads = df["Leads"].std()

                    if mean_leads != 0:
                        cv = (std_leads / mean_leads) * 100  # CV in percentage
                    else:
                        cv = 0  # Avoid division by zero

                    performance_consistency[sheet] = cv
                else:
                    performance_consistency[sheet] = 0

            # Find the associate with the lowest CV (most consistent)
            most_consistent = min(performance_consistency, key=performance_consistency.get)

            # Display the CVs in a dataframe
            consistency_df = pd.DataFrame({
                'Associate': list(performance_consistency.keys()),
                'Coefficient of Variation (%)': list(performance_consistency.values())
            }).sort_values('Coefficient of Variation (%)')

            st.dataframe(consistency_df)

            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(performance_consistency.keys(), performance_consistency.values(), color="lightskyblue")
            plt.axhline(performance_consistency[most_consistent], color="green", linestyle="--", 
                        label=f"Most Consistent: {most_consistent}")
            plt.title("Performance Consistency per Associate (CV %)")
            plt.xlabel("Associate")
            plt.ylabel("Coefficient of Variation (%)")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            st.success(f"🏅 Most Consistent Performer: **{most_consistent}** with CV = {performance_consistency[most_consistent]:.2f}%")
            st.markdown("""
            **Inference:**
            - The Coefficient of Variation (CV) measures the relative variability of lead generation performance.
            - Lower CV values indicate more consistent performance (less variation relative to the mean).
            - The associate with the lowest CV (green dashed line) has the most stable and predictable output.
            - Associates with high CV values have highly variable performance from day to day.
            - Consistent performance is valuable for predictable resource planning and reliable output.
            """)

        # ------ TAB 7: HIGH-PERFORMANCE DAYS ANALYSIS ------
        with tab7:
            st.header("High-Performance Days Analysis")

            # Dictionary to hold average time spent on high-performance days
            high_performance_time = {}

            # Process each associate
            for sheet in xls.sheet_names:
                df = sheets[sheet]

                if "Leads" in df.columns and "Time spent on LG (mins)" in df.columns:
                    df["Leads"] = pd.to_numeric(df["Leads"], errors="coerce").fillna(0)
                    df["Time spent on LG (mins)"] = pd.to_numeric(df["Time spent on LG (mins)"], errors="coerce").fillna(0)

                    # Find the 90th percentile threshold for leads
                    lead_threshold = df["Leads"].quantile(0.90)

                    # Filter top 10% high-performance days
                    high_perf_days = df[df["Leads"] >= lead_threshold]

                    if not high_perf_days.empty:
                        avg_time_high_perf = high_perf_days["Time spent on LG (mins)"].mean()
                    else:
                        avg_time_high_perf = 0

                    high_performance_time[sheet] = avg_time_high_perf
                else:
                    high_performance_time[sheet] = 0

            # Display the average times in a dataframe
            high_perf_df = pd.DataFrame({
                'Associate': list(high_performance_time.keys()),
                'Avg Time on High-Performance Days (mins)': list(high_performance_time.values())
            }).sort_values('Avg Time on High-Performance Days (mins)', ascending=False)

            st.dataframe(high_perf_df)

            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(high_performance_time.keys(), high_performance_time.values(), color="gold")
            plt.title("Average Time Spent on High-Performance Days per Associate")
            plt.xlabel("Associate")
            plt.ylabel("Average Time (mins)")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Find the associate with the highest and lowest time on high-performance days
            max_time_associate = max(high_performance_time, key=high_performance_time.get)
            min_time_associate = min(high_performance_time, key=high_performance_time.get)

            st.markdown(f"""
            **Inference:**
            - This analysis looks at how much time associates spend on lead generation during their best-performing days (top 10%).
            - **{max_time_associate}** spends the most time ({high_performance_time[max_time_associate]:.2f} mins) on high-performance days, suggesting their success may be effort-driven.
            - **{min_time_associate}** spends the least time ({high_performance_time[min_time_associate]:.2f} mins) while still achieving high performance, potentially indicating high efficiency.
            - Understanding the time investment on high-performance days can help identify optimal work patterns.
            - Some associates may achieve better results with focused, longer sessions, while others may be more effective with shorter, more efficient approaches.
            """)

            # Comparative Day Analysis: Weekdays vs Weekends
            st.subheader("Comparative Day Analysis: Weekdays vs Weekends")

            associate_selector_days = st.selectbox(
                "Select Associate for Day Analysis",
                options=xls.sheet_names,
                key="days_select"
            )

            df = sheets[associate_selector_days]

            # Data Cleaning
            if "Date" in df.columns and "Leads" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df["Leads"] = pd.to_numeric(df["Leads"], errors="coerce").fillna(0)

                df = df.dropna(subset=["Date"])  # Drop invalid dates

                if not df.empty:
                    # Identify if day is weekday (0-4) or weekend (5-6)
                    df["Day Type"] = df["Date"].dt.dayofweek.apply(
                        lambda x: "Weekend" if x >= 5 else "Weekday"
                    )

                    # Calculate average leads for Weekdays and Weekends
                    avg_leads = df.groupby("Day Type")["Leads"].mean()

                    weekday_avg = avg_leads.get("Weekday", 0)
                    weekend_avg = avg_leads.get("Weekend", 0)

                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 5))
                    avg_leads.reindex(["Weekday", "Weekend"]).plot(
                        kind="bar", color=["cornflowerblue", "lightcoral"], edgecolor="black", ax=ax
                    )
                    plt.title(f"Weekday vs Weekend Leads - {associate_selector_days}")
                    plt.xlabel("Day Type")
                    plt.ylabel("Average Leads")
                    plt.grid(axis="y", linestyle="--", alpha=0.7)
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)

                    diff = weekday_avg - weekend_avg
                    if diff > 0:
                        trend = "Better on Weekdays 📈"
                        st.success(f"**{associate_selector_days}** performs better on weekdays by {diff:.2f} leads on average.")
                    elif diff < 0:
                        trend = "Better on Weekends 🌞"
                        st.success(f"**{associate_selector_days}** performs better on weekends by {abs(diff):.2f} leads on average.")
                    else:
                        trend = "No major difference ⚖️"
                        st.info(f"**{associate_selector_days}** shows no significant difference between weekdays and weekends.")

                    st.markdown(f"""
                    **Day Analysis for {associate_selector_days}:**
                    - Weekday Average: {weekday_avg:.2f} leads
                    - Weekend Average: {weekend_avg:.2f} leads
                    - Performance Trend: {trend}
                    
                    **Inference:**
                    - This analysis helps understand if there's a pattern in performance based on the day of the week.
                    - Recognizing these patterns can help with optimal scheduling and resource allocation.
                    - Management can assign tasks that align with each associate's peak performance days.
                    """)

                else:
                    st.warning(f"Insufficient valid data for day analysis for {associate_selector_days}.")
            else:
                st.warning("Required columns missing for day analysis.")

        # ------ TAB 8: ADVANCED ANALYSIS ------
        with tab8:
            st.header("Predictive Model Analysis")

            # Associate selector for predictive model
            associate_selector_model = st.selectbox(
                "Select Associate for Predictive Analysis",
                options=xls.sheet_names,
                key="model_select"
            )

            df = sheets[associate_selector_model]

            # Data Cleaning
            if "Time spent on LG (mins)" in df.columns and "Leads" in df.columns:
                df["Time spent on LG (mins)"] = pd.to_numeric(df["Time spent on LG (mins)"], errors="coerce").fillna(0)
                df["Leads"] = pd.to_numeric(df["Leads"], errors="coerce").fillna(0)

                df = df.dropna(subset=["Time spent on LG (mins)", "Leads"])

                if not df.empty:
                    X = df[["Time spent on LG (mins)"]].values  # Feature
                    y = df["Leads"].values  # Target

                    # Linear Regression
                    model = LinearRegression()
                    model.fit(X, y)

                    # Predictions
                    y_pred = model.predict(X)

                    # Model accuracy (R-squared score)
                    r2_score = model.score(X, y)

                    # Visualization: Actual vs Predicted
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.scatter(X, y, color="dodgerblue", label="Actual Leads")
                    plt.plot(X, y_pred, color="red", linewidth=2, label="Predicted Leads")
                    plt.title(f"Actual vs Predicted Leads - {associate_selector_model}")
                    plt.xlabel("Time spent on LG (mins)")
                    plt.ylabel("Number of Leads")
                    plt.grid(alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Model performance
                    st.subheader("Model Performance")
                    st.info(f"Model R² Score: {r2_score:.4f}")

                    if r2_score > 0.7:
                        st.success("Strong predictive relationship between time spent and leads generated.")
                    elif r2_score > 0.4:
                        st.info("Moderate predictive relationship between time spent and leads generated.")
                    else:
                        st.warning("Weak predictive relationship between time spent and leads generated.")

                    # Prediction slider
                    st.subheader("Lead Prediction Tool")

                    time_input = st.slider(
                        "Select Time to Spend on Lead Generation (mins)",
                        min_value=0,
                        max_value=int(df["Time spent on LG (mins)"].max()) + 30,
                        value=60,
                        step=10
                    )

                    predicted_leads = float(model.predict([[time_input]])[0])
                    st.success(f"Predicted Leads for {time_input} minutes: {predicted_leads:.2f}")

                    # Inference
                    st.markdown("""
                    **Inference:**
                    - The scatter plot shows actual leads generated (blue dots) vs. the regression line prediction (red line).
                    - R² score indicates how well the model fits the data (0 = no fit, 1 = perfect fit).
                    - Use the slider to predict the expected number of leads based on the time investment.
                    - This predictive model can help optimize time allocation for lead generation activities.
                    """)

                    # Add correlation heatmap for multiple factors
                    st.subheader("Multi-factor Correlation Analysis")

                    # Select only numeric columns
                    numeric_df = df.select_dtypes(include=[np.number])

                    if not numeric_df.empty and numeric_df.shape[1] > 1:
                        # Calculate correlation matrix
                        corr_matrix = numeric_df.corr()

                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
                        plt.title(f"Correlation Matrix - {associate_selector_model}")
                        plt.tight_layout()
                        st.pyplot(fig)

                        st.markdown("""
                        **Correlation Matrix Interpretation:**
                        - This heatmap shows relationships between different numeric variables.
                        - Values close to 1 indicate strong positive correlation.
                        - Values close to -1 indicate strong negative correlation.
                        - Values near 0 indicate little to no correlation.
                        - Strong correlations can reveal key performance factors and optimization opportunities.
                        """)
                    else:
                        st.warning("Insufficient numeric data for correlation analysis.")

                else:
                    st.warning(f"Insufficient valid data for predictive analysis for {associate_selector_model}.")
            else:
                st.warning("Required columns missing for predictive analysis.")

            # Performance Benchmarking
            st.header("Team Performance Benchmarking")

            # Aggregate and compare performance metrics
            benchmark_data = {}

            for sheet in xls.sheet_names:
                df = sheets[sheet]

                if "Leads" in df.columns:
                    df["Leads"] = pd.to_numeric(df["Leads"], errors="coerce").fillna(0)

                    avg_leads = df["Leads"].mean()
                    max_leads = df["Leads"].max()

                    benchmark_data[sheet] = {
                        "Average Leads": avg_leads,
                        "Maximum Leads": max_leads
                    }

            # Convert to DataFrame for visualization
            benchmark_df = pd.DataFrame(benchmark_data).T

            # Calculate team average for benchmarking
            team_avg = benchmark_df["Average Leads"].mean()

            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            benchmark_df["Average Leads"].plot(kind="bar", color="lightblue", ax=ax)
            plt.axhline(y=team_avg, color='red', linestyle='--', label=f'Team Average: {team_avg:.2f}')
            plt.title("Team Performance Benchmarking")
            plt.xlabel("Associate")
            plt.ylabel("Average Leads")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(f"""
            **Team Performance Benchmarking Insights:**
            - The chart shows each associate's average lead generation compared to the team average ({team_avg:.2f} leads).
            - Associates above the red line are performing above the team average.
            - This benchmark provides a reference point for performance evaluation and target setting.
            - Management can identify top performers and those who may need additional support or training.
            """)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")


if __name__ == "__main__":
    main()
