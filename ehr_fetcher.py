import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional
from dotenv import load_dotenv
from fhir.resources.observation import Observation
from fhir.resources.bundle import Bundle
from pydantic import BaseModel, Field

load_dotenv()

class FallData(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    fall_date: str = Field(..., description="ISO date of fall")
    fall_time: str = Field(..., description="HH:MM time of fall")
    location: str = Field(default="Unknown", description="Fall location")
    cause: str = Field(default="Unknown", description="Cause of fall")
    injury: str = Field(default="None", description="Injury type")
    notes: str = Field(default="N/A", description="Additional notes")
    fall_status: int = Field(1, description="1 for fall, 0 for no-fall")

class EHRFetcher:
    def __init__(self):
        self.base_url = os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
        self.token = os.getenv("FHIR_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json"
        } if self.token else {"Accept": "application/fhir+json"}

    def fetch_falls(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        patient_ids: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[FallData]:
        """Fetch falls synchronously from FHIR."""
        return self._fetch_fhir(start_date, end_date, patient_ids, max_results)

    def _fetch_fhir(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        patient_ids: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[FallData]:
        """Fetch from FHIR server (synchronous with requests)."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).date().isoformat()
        if not end_date:
            end_date = datetime.now().date().isoformat()

        # Build params as a list so we can pass multiple 'date' entries (ge / le)
        params = [
            ("code", "161898005"),  # SNOMED for "Fall event"
            ("_count", str(max_results)),
            ("_sort", "-date"),
        ]
        if start_date:
            params.append(("date", f"ge{start_date}"))
        if end_date:
            params.append(("date", f"le{end_date}"))
        if patient_ids:
            for pid in patient_ids:
                params.append(("patient", pid))

        try:
            resp = requests.get(f"{self.base_url}/Observation", params=params, headers=self.headers, timeout=30)
            resp.raise_for_status()
            bundle = Bundle.parse_obj(resp.json())

            falls: List[FallData] = []
            for entry in bundle.entry or []:
                try:
                    obs = Observation.parse_obj(entry.resource)
                except Exception:
                    # skip malformed entries
                    continue

                # robust effectiveDateTime handling
                eff = getattr(obs, "effectiveDateTime", None)
                dt = None
                if eff:
                    try:
                        # eff may already be a string or FHIR type; coerce to str then parse
                        dt = datetime.fromisoformat(str(eff))
                    except Exception:
                        dt = None

                if dt:
                    fall_date = dt.date().isoformat()
                    fall_time = dt.strftime("%H:%M")
                else:
                    fall_date = start_date
                    fall_time = "00:00"

                # location (from extensions if present)
                location = "Unknown"
                try:
                    if getattr(obs, "extension", None):
                        ext0 = obs.extension[0]
                        location = getattr(ext0, "valueString", None) or location
                except Exception:
                    pass

                # cause (code.text or coding display)
                cause = "Unknown"
                try:
                    code = getattr(obs, "code", None)
                    if code:
                        cause = getattr(code, "text", None) or cause
                        if (not cause or cause == "Unknown") and getattr(code, "coding", None):
                            # try first coding display
                            coding0 = code.coding[0]
                            cause = getattr(coding0, "display", None) or cause
                except Exception:
                    pass

                # injury (valueString or a stringified value)
                injury = "None"
                try:
                    injury_val = getattr(obs, "valueString", None)
                    if injury_val:
                        injury = str(injury_val)
                    else:
                        # fallback to valueCodeableConcept or valueQuantity etc.
                        val_cc = getattr(obs, "valueCodeableConcept", None)
                        if val_cc:
                            # best-effort string
                            injury = getattr(val_cc, "text", None) or str(val_cc)
                except Exception:
                    injury = "None"

                # notes
                notes = "N/A"
                try:
                    if getattr(obs, "note", None):
                        notes = getattr(obs.note[0], "text", None) or notes
                except Exception:
                    notes = "N/A"

                fall = FallData(
                    patient_id=str(getattr(obs.subject, "reference", "Unknown")),
                    fall_date=fall_date,
                    fall_time=fall_time,
                    location=location or "Unknown",
                    cause=cause or "Unknown",
                    injury=injury or "None",
                    notes=notes or "N/A",
                    fall_status=1
                )
                falls.append(fall)

            # pagination: attempt to follow 'next' link if present and we need more
            next_url = next((link.url for link in (bundle.link or []) if getattr(link, "relation", None) == "next"), None)
            if next_url and len(falls) < max_results:
                resp2 = requests.get(next_url, headers=self.headers, timeout=30)
                resp2.raise_for_status()
                bundle2 = Bundle.parse_obj(resp2.json())
                for entry in bundle2.entry or []:
                    try:
                        obs = Observation.parse_obj(entry.resource)
                        # reuse same parsing logic as above (could refactor)
                        eff = getattr(obs, "effectiveDateTime", None)
                        dt = None
                        if eff:
                            try:
                                dt = datetime.fromisoformat(str(eff))
                            except Exception:
                                dt = None
                        if dt:
                            fall_date = dt.date().isoformat()
                            fall_time = dt.strftime("%H:%M")
                        else:
                            fall_date = start_date
                            fall_time = "00:00"
                        location = "Unknown"
                        try:
                            if getattr(obs, "extension", None):
                                ext0 = obs.extension[0]
                                location = getattr(ext0, "valueString", None) or location
                        except Exception:
                            pass
                        cause = "Unknown"
                        try:
                            code = getattr(obs, "code", None)
                            if code:
                                cause = getattr(code, "text", None) or cause
                                if (not cause or cause == "Unknown") and getattr(code, "coding", None):
                                    coding0 = code.coding[0]
                                    cause = getattr(coding0, "display", None) or cause
                        except Exception:
                            pass
                        injury = "None"
                        try:
                            injury_val = getattr(obs, "valueString", None)
                            if injury_val:
                                injury = str(injury_val)
                            else:
                                val_cc = getattr(obs, "valueCodeableConcept", None)
                                if val_cc:
                                    injury = getattr(val_cc, "text", None) or str(val_cc)
                        except Exception:
                            injury = "None"
                        notes = "N/A"
                        try:
                            if getattr(obs, "note", None):
                                notes = getattr(obs.note[0], "text", None) or notes
                        except Exception:
                            notes = "N/A"
                        fall = FallData(
                            patient_id=str(getattr(obs.subject, "reference", "Unknown")),
                            fall_date=fall_date,
                            fall_time=fall_time,
                            location=location or "Unknown",
                            cause=cause or "Unknown",
                            injury=injury or "None",
                            notes=notes or "N/A",
                            fall_status=1
                        )
                        falls.append(fall)
                    except Exception:
                        continue

            return falls
        except requests.exceptions.HTTPError as e:
            status = None
            try:
                status = e.response.status_code if e.response is not None else None
            except Exception:
                status = None
            if status == 429:
                import time
                time.sleep(60)
                return self._fetch_fhir(start_date, end_date, patient_ids, max_results)
            raise ValueError(f"FHIR API error: {status} - {getattr(e, 'response', '')}")
        except Exception as e:
            raise ValueError(f"Failed to fetch falls: {str(e)}")

if __name__ == "__main__":
    fetcher = EHRFetcher()
    falls = fetcher.fetch_falls(max_results=50)
    print(f"Fetched {len(falls)} falls")
    for fall in falls[:5]:
        print(fall.dict())
