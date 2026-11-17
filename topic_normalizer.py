
def normalize_topic(topic):

    """
    Convert similar AI-generated topics
    into standardized university categories.
    """

    topic = str(topic).strip().lower()

    # Teaching and Student Support
    teaching_keywords = [
        "teaching",
        "professor",
        "explanation",
        "student engagement",
        "helpfulness",
        "lecture speed",
        "lecture pace",
        "comprehensibility"
    ]

    # Exams and Assessment
    exam_keywords = [
        "exam",
        "assessment",
        "grading",
        "grade",
        "evaluation"
    ]

    # Learning Resources
    resource_keywords = [
        "course material",
        "learning resource",
        "lecture note",
        "outdated information",
        "study material"
    ]

    # Labs and Facilities
    lab_keywords = [
        "laboratory",
        "lab session",
        "lab equipment",
        "equipment",
        "facilities"
    ]

    # Workload
    workload_keywords = [
        "homework",
        "workload",
        "assignment",
        "study burden"
    ]

    # Course Experience
    experience_keywords = [
        "course enjoyment",
        "course experience",
        "course quality",
        "course structure",
        "course organization"
    ]

    if any(word in topic for word in teaching_keywords):
        return "Teaching & Student Support"

    elif any(word in topic for word in exam_keywords):
        return "Exams & Assessment"

    elif any(word in topic for word in resource_keywords):
        return "Learning Resources"

    elif any(word in topic for word in lab_keywords):
        return "Labs & Facilities"

    elif any(word in topic for word in workload_keywords):
        return "Workload"

    elif any(word in topic for word in experience_keywords):
        return "Course Experience"

    else:
        return "Other"


def normalize_topics(topics):

    """
    Normalize a list of topics and
    remove duplicate categories.
    """

    if not isinstance(topics, list):
        return ["Other"]

    normalized = []

    for topic in topics:

        category = normalize_topic(topic)

        if category not in normalized:
            normalized.append(category)

    return normalized
