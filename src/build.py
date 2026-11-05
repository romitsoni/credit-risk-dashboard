"""Runs the full pipeline: clean data -> train model -> flag anomalies -> insights."""
import ingest
import model
import anomaly
import analytics

if __name__ == "__main__":
    print("== 1/4 Ingest ==")
    ingest.build()
    print("\n== 2/4 Model ==")
    model.main()
    print("\n== 3/4 Anomalies ==")
    anomaly.main()
    print("\n== 4/4 Insights ==")
    analytics.main()
    print("\nDone. Start the app with: streamlit run app/app.py")