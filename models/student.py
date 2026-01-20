from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Student:
    """Lightweight student record used by both models."""

    id: int
    gender: str = ""
    race_ethnicity: str = ""
    parental_level_of_education: str = ""
    lunch: str = ""
    test_preparation_course: str = ""
    math_score: int = 0
    reading_score: int = 0
    writing_score: int = 0

    @classmethod
    def from_csv_row(cls, row: dict, *, id_value: int) -> "Student":
        """Create a student from a CSV row matching StudentsPerformance schema."""
        race = row.get("race_ethnicity") or row.get("race/ethnicity", "")
        parental = row.get("parental_level_of_education") or row.get("parental level of education", "")
        lunch = row.get("lunch", "")
        prep = row.get("test_preparation_course") or row.get("test preparation course", "")
        math = row.get("math_score") or row.get("math score", 0)
        reading = row.get("reading_score") or row.get("reading score", 0)
        writing = row.get("writing_score") or row.get("writing score", 0)
        return cls(
            id=id_value,
            gender=row.get("gender", "").strip(),
            race_ethnicity=race.strip(),
            parental_level_of_education=parental.strip(),
            lunch=lunch.strip(),
            test_preparation_course=prep.strip(),
            math_score=int(math or 0),
            reading_score=int(reading or 0),
            writing_score=int(writing or 0),
        )

    def to_summary(self) -> str:
        return (
            f"id={self.id} gender={self.gender} race={self.race_ethnicity} "
            f"math={self.math_score} reading={self.reading_score} writing={self.writing_score}"
        )
