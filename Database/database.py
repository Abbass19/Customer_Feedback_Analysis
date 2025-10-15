import psycopg2
from psycopg2 import sql
from typing import Optional, Dict, List, Any


# ----- 1. Database connection -----
def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="feedback_db",
        user="postgres",
        password="NewPassword2004"
    )

# ----- 2. Insert function -----
def DB_ADD_Record(feedback_text: str,
                  sentiment_scores: Dict[str, Optional[int]],
                  ner_values: Dict[str, Optional[str]]):
    """
    Inserts a processed feedback record into the hospital_feedback table.

    :param feedback_text: raw feedback text
    :param sentiment_scores: dict with keys: ['pricing','appointments','staff','customer_service','emergency_services']
    :param ner_values: dict with NER keys matching the table, can be None if not present
    """
    query = sql.SQL("""
        INSERT INTO hospital_feedback (
            feedback_text,
            sentiment_pricing,
            sentiment_appointments,
            sentiment_staff,
            sentiment_customer_service,
            sentiment_emergency_services,
            doctor_name,
            staff_role,
            hospital_name,
            department,
            specialty,
            service_area,
            price,
            time_expression,
            location,
            quality_aspect,
            issue_type,
            treatment_type
        ) VALUES (
            %(feedback_text)s,
            %(sentiment_pricing)s,
            %(sentiment_appointments)s,
            %(sentiment_staff)s,
            %(sentiment_customer_service)s,
            %(sentiment_emergency_services)s,
            %(doctor_name)s,
            %(staff_role)s,
            %(hospital_name)s,
            %(department)s,
            %(specialty)s,
            %(service_area)s,
            %(price)s,
            %(time_expression)s,
            %(location)s,
            %(quality_aspect)s,
            %(issue_type)s,
            %(treatment_type)s
        )
    """)

    # Combine all values
    data = {
        "feedback_text": feedback_text,
        "sentiment_pricing": sentiment_scores.get("pricing"),
        "sentiment_appointments": sentiment_scores.get("appointments"),
        "sentiment_staff": sentiment_scores.get("staff"),
        "sentiment_customer_service": sentiment_scores.get("customer_service"),
        "sentiment_emergency_services": sentiment_scores.get("emergency_services"),
        **{key: ner_values.get(key) for key in [
            "doctor_name", "staff_role", "hospital_name", "department", "specialty",
            "service_area", "price", "time_expression", "location", "quality_aspect",
            "issue_type", "treatment_type"
        ]}
    }

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, data)
        conn.commit()
        print("Feedback inserted successfully!")
    except Exception as e:
        conn.rollback()
        print("Error inserting feedback:", e)
    finally:
        conn.close()



def get_all_feedback(limit: Optional[int] = None) -> List[Dict]:
    """
    Returns all feedback records from the hospital_feedback table.
    Optional limit to avoid huge responses.
    """
    conn = get_connection()
    results = []
    try:
        with conn.cursor() as cur:
            query = "SELECT * FROM hospital_feedback"
            if limit:
                query += f" LIMIT {limit}"
            cur.execute(query)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            for row in rows:
                results.append(dict(zip(colnames, row)))
    finally:
        conn.close()
    return results

# ----- 3. Retrieve feedback by ID -----
def get_feedback_by_id(record_id: int) -> Optional[Dict]:
    """
    Returns a single feedback record by its ID.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM hospital_feedback WHERE id = %s",
                (record_id,)
            )
            row = cur.fetchone()
            if row:
                colnames = [desc[0] for desc in cur.description]
                return dict(zip(colnames, row))
            else:
                return None
    finally:
        conn.close()

# Returns records where the specified NER fields (indices 0–12) are not null.
def get_records_with_ner(selected_indices: List[int]) -> List[Dict[str, Any]]:
    NER_COLUMNS = [
        "doctor_name",
        "staff_role",
        "hospital_name",
        "department",
        "specialty",
        "service_area",
        "price",
        "feedback_text",
        "time_expression",
        "location",
        "quality_aspect",
        "issue_type",
        "treatment_type"
    ]

    if not selected_indices:
        return []

    # Build SQL WHERE clause to exclude both NULL and empty strings
    conditions = [f"{NER_COLUMNS[i]} IS NOT NULL AND {NER_COLUMNS[i]} <> ''" for i in selected_indices]
    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT id, {', '.join(NER_COLUMNS)}, created_at
        FROM hospital_feedback
        WHERE {where_clause}
        ORDER BY created_at DESC;
    """

    conn = get_connection()
    results = []
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            for row in rows:
                results.append(dict(zip(colnames, row)))
    finally:
        conn.close()

    return results

# Returns records matching only the specified sentiment values for the given sentiment columns
def get_records_with_sentiment(sentiment_filter: Dict[str, List[int]]) -> List[Dict[str, Any]]:
    """
    Return records filtered by the specified sentiment values.

    :param sentiment_filter: dict where key is a sentiment column name and value is a list of allowed integers (0-3)
                             e.g., {"sentiment_pricing": [2,3], "sentiment_staff": [1,2]}
    :return: list of dicts representing the records
    """
    SENTIMENT_COLUMNS = [
        "sentiment_pricing",
        "sentiment_appointments",
        "sentiment_staff",
        "sentiment_customer_service",
        "sentiment_emergency_services"
    ]

    # Filter only valid columns
    valid_filter = {k: v for k, v in sentiment_filter.items() if k in SENTIMENT_COLUMNS and v}
    if not valid_filter:
        return []  # no valid filter provided

    # Build SQL WHERE clause
    conditions = [f"{col} IN %s" for col in valid_filter.keys()]
    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT id, {', '.join(SENTIMENT_COLUMNS)}, feedback_text, created_at
        FROM hospital_feedback
        WHERE {where_clause}
        ORDER BY created_at DESC;
    """

    conn = get_connection()
    results = []
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple([tuple(vals) for vals in valid_filter.values()]))
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            for row in rows:
                results.append(dict(zip(colnames, row)))
    finally:
        conn.close()

    return results

# Delete a single record by its ID
def delete_record_by_id(record_id: int) -> bool:
    """Delete a single record from hospital_feedback by ID."""
    query = "DELETE FROM hospital_feedback WHERE id = %s;"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (record_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print("Error deleting record:", e)
        return False
    finally:
        conn.close()

# Delete all records in the table
def clear_all_records() -> bool:
    """Delete all records from hospital_feedback."""
    query = "TRUNCATE TABLE hospital_feedback RESTART IDENTITY;"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print("Error clearing records:", e)
        return False
    finally:
        conn.close()

# ----- Print schema function -----
def print_table_schema(table_name: str):
    """
    Prints the schema (columns and types) of the given table.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name=%s;
            """, (table_name,))
            columns = cur.fetchall()
            print(f"Schema for table '{table_name}':")
            for col_name, col_type in columns:
                print(f" {col_name} : {col_type}")
    except Exception as e:
        print("Error fetching schema:", e)
    finally:
        conn.close()



if __name__ == "__main__":
    # print_table_schema("hospital_feedback")
    # print(get_feedback_by_id(1))
    # print(get_all_feedback())

    # Get records with doctor_name and hospital_name not null
    # records = get_records_with_ner([0,1])
    # for rec in records:
    #     print(rec)

    # filter_dict = {
    #     "sentiment_staff": [1],
    #     "sentiment_customer_service": [1]
    # }
    #
    # records = get_records_with_sentiment(filter_dict)
    # for r in records:
    #     print(r)
    #
    #
    delete_record_by_id(6)


