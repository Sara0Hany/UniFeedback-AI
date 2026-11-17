
import streamlit as st
import pandas as pd
import plotly.express as px

from ai_analyzer import analyze_reviews
from topic_normalizer import normalize_topics
from report_generator import generate_executive_report


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="UniFeedback AI",
    page_icon="🎓",
    layout="wide"
)

with st.sidebar:
    st.title("⚙️ AI Control Panel")
    st.info("Model: Qwen 2.5 3B (Local Ollama)")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=300,
        max_value=1500,
        value=500,
        step=100
    )

    st.write(f"Temperature: {temperature}")
    st.write(f"Max Tokens: {max_tokens}")


# ==========================================
# 2. APPLICATION HEADER
# ==========================================

st.title("🎓 UniFeedback AI")

st.subheader(
    "University Feedback Intelligence Platform"
)

st.write(
    "Analyze student feedback using Local Generative AI "
    "and discover insights to improve education."
)

st.divider()


# ==========================================
# 3. FILE UPLOAD
# ==========================================

st.header("📁 Upload Student Feedback")

uploaded_file = st.file_uploader(
    "Choose your CSV or Excel file",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    try:

        # Read uploaded file
        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

    except pd.errors.EmptyDataError:

        st.error("The uploaded CSV file is empty.")
        st.stop()

    except Exception as e:

        st.error(f"Error reading file: {e}")
        st.stop()


    # ==========================================
    # 4. VALIDATE DATA
    # ==========================================

    review_column = next(
        (
            col for col in df.columns
            if str(col).strip().lower() == "review"
        ),
        None
    )

    if review_column is None:

        st.error(
            "Invalid dataset. Your file must contain "
            "a column named 'review'."
        )

        st.stop()


    # ==========================================
    # 5. DATA CLEANING
    # ==========================================

    df = df.rename(
        columns={review_column: "review"}
    )

    df["review"] = (
        df["review"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty reviews
    df = df[df["review"] != ""]

    # Remove duplicates
    df = df.drop_duplicates(
        subset=["review"]
    ).reset_index(drop=True)


    if df.empty:

        st.warning(
            "No valid student reviews were found."
        )

        st.stop()


    # Clear previous results when a different file is uploaded
    current_file_id = (
        f"{uploaded_file.name}_{uploaded_file.size}"
    )

    if st.session_state.get("current_file_id") != current_file_id:

        st.session_state.pop("analysis_results", None)
        st.session_state.pop("executive_report", None)

        st.session_state["current_file_id"] = current_file_id


    st.session_state["reviews"] = df


    # ==========================================
    # 6. DATASET INFORMATION
    # ==========================================

    st.success("File uploaded successfully!")

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Reviews",
        len(df)
    )

    col2.metric(
        "Total Columns",
        len(df.columns)
    )

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()


    # ==========================================
    # 7. AI ANALYSIS
    # ==========================================

    st.header("🤖 AI Feedback Analysis")

    st.write(
        "Analyze all student reviews using "
        "the local Qwen Generative AI model."
    )

    if st.button(
        "🚀 Analyze All Reviews",
        type="primary",
        use_container_width=True
    ):

        progress_bar = st.progress(0)

        status = st.empty()

        status.info(
            "AI is analyzing student feedback..."
        )

        try:
            st.session_state.pop("executive_report", None)

            # Run AI analysis
            results = analyze_reviews(
    df["review"].tolist(),
    progress_callback=progress_bar.progress,
    temperature=temperature,
    max_tokens=max_tokens
)

            # Save results
            st.session_state["analysis_results"] = (
                pd.DataFrame(results)
            )

            progress_bar.progress(100)

            status.success(
                "All reviews have been processed!"
            )

        except Exception as e:

            status.error(
                f"Analysis failed: {e}"
            )


    # ==========================================
    # 8. DISPLAY ANALYSIS RESULTS
    # ==========================================

    if "analysis_results" in st.session_state:

        results_df = st.session_state[
            "analysis_results"
        ].copy()

        st.divider()

        st.header("📊 Feedback Analytics Dashboard")


        # Separate successful and failed results
        valid_df = results_df[
            results_df["error"] == ""
        ].copy()

        failed_df = results_df[
            results_df["error"] != ""
        ].copy()

        # Show clear Ollama connection errors

        if not failed_df.empty:

            connection_errors = failed_df[
                failed_df["error"].str.contains(
                    "Ollama|connect",
                    case=False,
                    na=False
                )
            ]

            if not connection_errors.empty:

                st.error(
                    "Ollama Connection Error: "
                    "Cannot connect to the local AI model. "
                    "Please start Ollama and try again."
                )

            else:

                st.warning(
                    f"{len(failed_df)} reviews failed. "
                    "Check the Failed Reviews section for details."
                )





        # ==========================================
        # 9. NORMALIZE TOPICS
        # ==========================================

        if not valid_df.empty:

            valid_df["normalized_topics"] = (
                valid_df["topics"].apply(normalize_topics)
            )

        else:

            valid_df["normalized_topics"] = []


        # ==========================================
        # 10. KEY PERFORMANCE INDICATORS
        # ==========================================

        st.subheader("📌 Analysis Overview")

        total_reviews = len(results_df)

        successful_reviews = len(valid_df)

        failed_reviews = len(failed_df)

        positive = (
            valid_df["sentiment"] == "Positive"
        ).sum()

        negative = (
            valid_df["sentiment"] == "Negative"
        ).sum()

        neutral = (
            valid_df["sentiment"] == "Neutral"
        ).sum()

        mixed = (
            valid_df["sentiment"] == "Mixed"
        ).sum()


        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Reviews",
            total_reviews
        )

        col2.metric(
            "Successfully Analyzed",
            successful_reviews
        )

        col3.metric(
            "Positive Reviews",
            int(positive)
        )

        col4.metric(
            "Negative Reviews",
            int(negative)
        )


        col5, col6, col7 = st.columns(3)

        col5.metric(
            "Neutral Reviews",
            int(neutral)
        )

        col6.metric(
            "Mixed Reviews",
            int(mixed)
        )

        col7.metric(
            "Failed Reviews",
            failed_reviews
        )


        st.divider()


        # ==========================================
        # 11. SENTIMENT DISTRIBUTION
        # ==========================================

        st.subheader("😊 Sentiment Distribution")

        if not valid_df.empty:

            sentiment_counts = (
                valid_df["sentiment"]
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Count")
            )

            fig_sentiment = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                hole=0.45,
                title="Student Feedback Sentiment"
            )

            fig_sentiment.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )

            st.plotly_chart(
                fig_sentiment,
                use_container_width=True
            )

        else:

            st.info("No sentiment data available.")


        st.divider()


        # ==========================================
        # 12. STANDARDIZED TOPIC FREQUENCY
        # ==========================================

        st.subheader("📚 Most Discussed Topics")

        if not valid_df.empty:

            topics_df = valid_df[
                ["normalized_topics"]
            ].explode("normalized_topics")

            topics_df = topics_df.dropna()

            topics_df = topics_df[
                topics_df["normalized_topics"].astype(str).str.strip() != ""
            ]

            topic_counts = (
                topics_df["normalized_topics"]
                .value_counts()
                .reset_index()
            )

            topic_counts.columns = [
                "Topic",
                "Number of Reviews"
            ]

            if not topic_counts.empty:

                fig_topics = px.bar(
                    topic_counts,
                    x="Number of Reviews",
                    y="Topic",
                    orientation="h",
                    title="Standardized University Feedback Topics",
                    text="Number of Reviews"
                )

                fig_topics.update_layout(
                    yaxis={
                        "categoryorder": "total ascending"
                    }
                )

                st.plotly_chart(
                    fig_topics,
                    use_container_width=True
                )

                # Show topic frequency table
                with st.expander("View Topic Frequency Table"):

                    st.dataframe(
                        topic_counts,
                        use_container_width=True,
                        hide_index=True
                    )

            else:

                st.info("No topics were detected.")


        st.divider()


        # ==========================================
        # 13. TOPICS BY SENTIMENT
        # ==========================================

        st.subheader("🔍 Topics by Sentiment")

        if not valid_df.empty:

            topic_sentiment = valid_df.explode(
                "normalized_topics"
            )

            topic_sentiment = topic_sentiment.dropna(
                subset=["normalized_topics"]
            )

            topic_sentiment = topic_sentiment[
                topic_sentiment["normalized_topics"].astype(str).str.strip() != ""
            ]

            if not topic_sentiment.empty:

                cross_data = pd.crosstab(
                    topic_sentiment["normalized_topics"],
                    topic_sentiment["sentiment"]
                )

                sentiment_order = [
                    "Positive",
                    "Negative",
                    "Neutral",
                    "Mixed"
                ]

                cross_data = cross_data.reindex(
                    columns=sentiment_order,
                    fill_value=0
                )

                fig_cross = px.bar(
                    cross_data.reset_index(),
                    x="normalized_topics",
                    y=sentiment_order,
                    barmode="group",
                    title="Sentiment Comparison by Standardized Topic",
                    labels={
                        "normalized_topics": "Topic",
                        "value": "Number of Reviews",
                        "variable": "Sentiment"
                    }
                )

                st.plotly_chart(
                    fig_cross,
                    use_container_width=True
                )

            else:

                st.info(
                    "No topic-sentiment data available."
                )


        st.divider()

 
        #
        # ==========================================
        # FILTERED REVIEWS TABLE
        # ==========================================

        st.subheader("🔎 Filtered Student Reviews")

        if not valid_df.empty:

            filtered_display = valid_df[
                [
                    "review",
                    "sentiment",
                    "normalized_topics",
                    "main_issue",
                    "recommendation"
                ]
            ].copy()

            filtered_display["normalized_topics"] = (
                filtered_display["normalized_topics"].apply(
                    lambda x: ", ".join(x)
                    if isinstance(x, list)
                    else str(x)
                )
            )

            st.dataframe(
                filtered_display,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No reviews match the selected filters."
            )

        #  ==========================================
        # 14. FULL AI ANALYSIS RESULTS
        # ==========================================

        st.subheader("📋 Detailed AI Analysis Results")

        with st.expander(
            "View Original AI Topics and Standardized Topics"
        ):

            display_df = valid_df.copy()

            display_df["topics"] = (
                display_df["topics"].apply(
                    lambda x: ", ".join(x)
                    if isinstance(x, list)
                    else str(x)
                )
            )

            display_df["normalized_topics"] = (
                display_df["normalized_topics"].apply(
                    lambda x: ", ".join(x)
                    if isinstance(x, list)
                    else str(x)
                )
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


        # ==========================================
        # 15. FAILED REVIEWS
        # ==========================================

        if not failed_df.empty:

            st.subheader("⚠️ Failed Reviews")

            st.dataframe(
                failed_df,
                use_container_width=True
            )


        st.divider()


        # ==========================================
        # 16. EXPORT RESULTS
        # ==========================================

        st.header("📥 Export Analysis Results")

        export_df = valid_df.copy()

        # Convert topic lists into text
        export_df["topics"] = (
            export_df["topics"].apply(
                lambda x: ", ".join(x)
                if isinstance(x, list)
                else str(x)
            )
        )

        export_df["normalized_topics"] = (
            export_df["normalized_topics"].apply(
                lambda x: ", ".join(x)
                if isinstance(x, list)
                else str(x)
            )
        )

        csv_data = export_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download Complete Analysis (CSV)",
            data=csv_data,
            file_name="unifeedback_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )


    # ==========================================
    # 17. AI EXECUTIVE REPORT
    # ==========================================

    if "analysis_results" in st.session_state:

        st.divider()

        st.header("🧠 AI Executive Summary")

        st.write(
            "Generate an intelligent management report "
            "based on the analyzed student feedback."
        )

        if st.button(
            "✨ Generate Executive Report",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "AI is generating the executive report..."
            ):

                try:
                    report_df = st.session_state[
                        "analysis_results"
                    ].copy()

                    report_df["normalized_topics"] = (
                        report_df["topics"].apply(
                            normalize_topics
                        )
                    )

                    report = generate_executive_report(
                        report_df
                    )

                    st.session_state[
                        "executive_report"
                    ] = report

                    st.success(
                        "Executive report generated!"
                    )

                except Exception as e:

                    st.error(
                        f"Report generation failed: {e}"
                    )

        # Display generated report
        if "executive_report" in st.session_state:

            report = st.session_state[
                "executive_report"
            ]

            st.subheader("📌 Overall Assessment")

            st.write(
                report["overall_assessment"]
            )

            st.subheader("✅ Main Strengths")

            for item in report["main_strengths"]:
                st.write(f"✅ {item}")

            st.subheader("⚠️ Critical Issues")

            for item in report["critical_issues"]:
                st.write(f"⚠️ {item}")

            st.subheader("💡 Smart Recommendations")

            for index, item in enumerate(
                report["recommendations"], start=1
            ):
                st.write(f"{index}. {item}")

            st.subheader("🎯 Priority Action")

            st.info(
                report["priority_action"]
            )

            # Create downloadable text report
            report_text = (
                "UNIFEEDBACK AI - EXECUTIVE REPORT\n\n"
                "OVERALL ASSESSMENT\n"
                f"{report['overall_assessment']}\n\n"
                "MAIN STRENGTHS\n"
                + "\n".join(
                    f"- {x}" for x in report["main_strengths"]
                )
                + "\n\nCRITICAL ISSUES\n"
                + "\n".join(
                    f"- {x}" for x in report["critical_issues"]
                )
                + "\n\nRECOMMENDATIONS\n"
                + "\n".join(
                    f"- {x}" for x in report["recommendations"]
                )
                + "\n\nPRIORITY ACTION\n"
                + report["priority_action"]
            )

            st.download_button(
                label="📥 Download Executive Report",
                data=report_text.encode("utf-8"),
                file_name="unifeedback_report.txt",
                mime="text/plain",
                use_container_width=True
            )

else:

    st.info(
        "Please upload a CSV or Excel file to begin."
    )

    st.markdown(
        """
        ### How to use UniFeedback AI

        1. Upload your student feedback dataset.
        2. Review the uploaded data.
        3. Click Analyze All Reviews.
        4. Explore sentiment and topic analytics.
        5. Download your analysis results.
        """
    )
