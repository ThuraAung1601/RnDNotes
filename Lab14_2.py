class University():
    def __init__(self, name, programs):
        self.name = name
        self.programs = programs  # list of Program objects
        self.students = {}  # key = ID, value = Student object

    def addStudent(self, student):
        self.students[student.id] = student

    def getStudent(self, student_id):
        return self.students.get(student_id, None)


class Program():
    def __init__(self, level: str, name: str, start: str, courses: list):
        self.level = level
        self.name = name
        self.start = start
        self.courses = courses  # list of Course objects

    def addCourse(self, course):
        self.courses.append(course)

    def getCourse(self, course_id):
        for course in self.courses:
            if course.id == course_id:
                return course
        return None


class Course():
    def __init__(self, credit: int, id: int, lecturer, name: str, semester: str, student_list: list):
        self.credit = credit
        self.id = id
        self.lecturer = lecturer  # Lecturer object
        self.name = name
        self.semester = semester
        self.student_list = student_list  # list of Student objects

    def enroll(self, student):
        if student not in self.student_list:
            self.student_list.append(student)
            student.course_list.append(self)

    def getCredit(self):
        return self.credit

    def getLecturer(self):
        return self.lecturer

    def getStudents(self):
        return self.student_list


class Lecturer():
    def __init__(self, name: str, id: int, course_list: list):
        self.name = name
        self.id = id
        self.course_list = course_list  # list of Course objects

    def getCourse(self, course_id):
        for course in self.course_list:
            if course.id == course_id:
                return course
        return None

    def addCourse(self, course):
        self.course_list.append(course)


class Student():
    def __init__(self, id: int, name: str, course_list: list = None):
        if course_list is None:
            course_list = []
        self.name = name
        self.id = id
        self.status = "normal"
        self.course_list = course_list  # list of Course objects

    def enrollCourse(self, course):
        if course not in self.course_list:
            self.course_list.append(course)
            course.enroll(self)


class Take():
    def __init__(self, grade: str, scores: int, student, course):
        self.grade = grade
        self.scores = scores
        self.student = student  # Student object
        self.course = course  # Course object


class Transcript():
    def __init__(self, complete: bool, issue_date: str, take_list: list):
        self.complete = complete
        self.issue_date = issue_date
        self.take_list = take_list  # list of Take objects

    def printTranscript(self):
        print(f"Transcript Issued on: {self.issue_date}")
        print(f"Status: {'Complete' if self.complete else 'Incomplete'}")
        print("Courses Taken:")
        for take in self.take_list:
            print(f"- {take.course.name} ({take.course.id}): Grade = {take.grade}, Score = {take.scores}")


if __name__ == "__main__":
    # Create Lecturer
    lecturer1 = Lecturer(name="Dr. John Smith", id=101, course_list=[])

    # Create Courses
    course1 = Course(credit=3, id=201, lecturer=lecturer1, name="Algorithms", semester="Fall", student_list=[])
    course2 = Course(credit=4, id=202, lecturer=lecturer1, name="Data Structures", semester="Fall", student_list=[])

    # Add Courses to Lecturer
    lecturer1.addCourse(course1)
    lecturer1.addCourse(course2)

    # Create Program and add Courses
    program1 = Program(level="Bachelor", name="Computer Science", start="2024", courses=[])
    program1.addCourse(course1)
    program1.addCourse(course2)

    # Create University and add Program
    university = University(name="Global Tech University", programs=[])
    university.programs.append(program1)

    # Create Student
    student1 = Student(id=301, name="Alice Johnson")
    university.addStudent(student1)

    # Enroll Student in Courses
    student1.enrollCourse(course1)
    student1.enrollCourse(course2)

    # Show enrolled students in a course
    print("\nStudents in Algorithms course:")
    for s in course1.getStudents():
        print(f"- {s.name}")

    # Create Take instances (grade and score)
    take1 = Take(grade="A", scores=90, student=student1, course=course1)
    take2 = Take(grade="B+", scores=85, student=student1, course=course2)

    # Generate Transcript
    transcript = Transcript(complete=True, issue_date="2025-03-14", take_list=[take1, take2])

    # Print Transcript
    print("\nTranscript for Alice Johnson:")
    transcript.printTranscript()
