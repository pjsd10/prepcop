"""Durable operation ledger alongside LangGraph's SQLite checkpoints."""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from .errors import BudgetExceeded, UnknownOperation

def utc_now():
    return datetime.now(timezone.utc).isoformat()

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript('''
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, created TEXT NOT NULL,
                counters TEXT NOT NULL, elapsed REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS operations(run_id TEXT, id TEXT, kind TEXT,
                status TEXT, arguments TEXT, result TEXT, started REAL, timeout REAL,
                PRIMARY KEY(run_id,id));
            CREATE TABLE IF NOT EXISTS events(n INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT, at TEXT, kind TEXT, details TEXT);
            CREATE TABLE IF NOT EXISTS readable(run_id TEXT, source_id TEXT,
                topic_id TEXT, PRIMARY KEY(run_id,source_id));
            CREATE TABLE IF NOT EXISTS submissions(run_id TEXT, id TEXT, status TEXT,
                PRIMARY KEY(run_id,id));
            CREATE TABLE IF NOT EXISTS allowances(run_id TEXT, kind TEXT, operation_id TEXT,
                PRIMARY KEY(run_id,kind));
            ''')

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def create(self, run_id):
        counters = dict(model_attempts=0, search_attempts=0, extract_attempts={},
                        readable_pages={}, transient_retries=0, repairs=0)
        with self.connect() as con:
            con.execute("INSERT INTO runs(id,created,counters) VALUES(?,?,?)", (run_id,utc_now(),json.dumps(counters)))

    def counters(self, run_id):
        with self.connect() as con:
            row = con.execute("SELECT counters,elapsed FROM runs WHERE id=?",(run_id,)).fetchone()
        if not row:
            raise ValueError("Saved run not found. Check the complete resume code.")
        c = json.loads(row["counters"])
        c["research_elapsed_seconds"] = row["elapsed"]
        return c

    def charge(self, run_id, kind, topic_id=None):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT counters FROM runs WHERE id=?",(run_id,)).fetchone()
            c = json.loads(row[0])
            if kind == "model":
                if c["model_attempts"] >= 32: raise BudgetExceeded("The 32 model-attempt limit was reached.")
                c["model_attempts"] += 1
            elif kind == "search":
                if c["search_attempts"] >= 4: raise BudgetExceeded("The four-search limit was reached.")
                c["search_attempts"] += 1
            elif kind == "extract":
                if c["extract_attempts"].get(topic_id,0) >= 5: raise BudgetExceeded("The topic extraction limit was reached.")
                c["extract_attempts"][topic_id] = c["extract_attempts"].get(topic_id,0)+1
            elif kind in ("repairs", "transient_retries"):
                if c[kind] >= 1: return False
                c[kind] += 1
            else: raise ValueError("Unknown budget counter")
            con.execute("UPDATE runs SET counters=? WHERE id=?",(json.dumps(c),run_id))
        return True

    def add_elapsed(self, run_id, seconds):
        with self.connect() as con:
            con.execute("UPDATE runs SET elapsed=elapsed+? WHERE id=?",(max(0,seconds),run_id))

    def readable(self, run_id, source_id, topic_id):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if con.execute("SELECT 1 FROM readable WHERE run_id=? AND source_id=?",(run_id,source_id)).fetchone(): return
            c=json.loads(con.execute("SELECT counters FROM runs WHERE id=?",(run_id,)).fetchone()[0])
            if c["readable_pages"].get(topic_id,0)>=4: raise BudgetExceeded("The topic readable-page limit was reached.")
            c["readable_pages"][topic_id]=c["readable_pages"].get(topic_id,0)+1
            con.execute("INSERT INTO readable VALUES(?,?,?)",(run_id,source_id,topic_id))
            con.execute("UPDATE runs SET counters=? WHERE id=?",(json.dumps(c),run_id))

    def operation(self, run_id, key):
        with self.connect() as con:
            row=con.execute("SELECT * FROM operations WHERE run_id=? AND id=?",(run_id,key)).fetchone()
        return dict(row) if row else None

    def begin_operation(self, run_id, key, kind, arguments, timeout):
        with self.connect() as con:
            con.execute("INSERT INTO operations VALUES(?,?,?,?,?,?,?,?)",(run_id,key,kind,"pending",json.dumps(arguments),None,time.time(),timeout))

    def finish_operation(self, run_id, key, result, status="completed", elapsed=0):
        with self.connect() as con:
            con.execute("UPDATE operations SET status=?,result=? WHERE run_id=? AND id=?",(status,json.dumps(result),run_id,key))
            if elapsed:
                con.execute("UPDATE runs SET elapsed=elapsed+? WHERE id=?",(max(0,elapsed),run_id))

    def unresolved(self, run_id):
        with self.connect() as con:
            rows=con.execute("SELECT id,kind,status FROM operations WHERE run_id=? AND status IN ('pending','unknown')",(run_id,)).fetchall()
        return [dict(r) for r in rows]

    def failed(self, run_id):
        with self.connect() as con:
            rows=con.execute("SELECT id FROM operations WHERE run_id=? AND status='failed'",(run_id,)).fetchall()
        return [dict(r) for r in rows]

    def has_submission(self, run_id, event_id):
        with self.connect() as con:
            return bool(con.execute("SELECT 1 FROM submissions WHERE run_id=? AND id=?",(run_id,event_id)).fetchone())

    def allowance(self, run_id, kind, operation_id):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            old=con.execute("SELECT operation_id FROM allowances WHERE run_id=? AND kind=?",(run_id,kind)).fetchone()
            if old: return old[0] == operation_id
            c=json.loads(con.execute("SELECT counters FROM runs WHERE id=?",(run_id,)).fetchone()[0])
            if c[kind]>=1: return False
            c[kind]+=1
            con.execute("UPDATE runs SET counters=? WHERE id=?",(json.dumps(c),run_id))
            con.execute("INSERT INTO allowances VALUES(?,?,?)",(run_id,kind,operation_id))
            return True

    def allowance_owner(self, run_id, kind):
        with self.connect() as con:
            row=con.execute("SELECT operation_id FROM allowances WHERE run_id=? AND kind=?",(run_id,kind)).fetchone()
        return row[0] if row else None

    def resolve_unknown_time(self, run_id, key):
        # A crashed process cannot save its stopwatch; charge the reserved maximum
        # once. A caught wait timeout has already saved elapsed time.
        with self.connect() as con:
            row=con.execute("SELECT * FROM operations WHERE run_id=? AND id=?",(run_id,key)).fetchone()
            if row and row["status"] == "pending":
                if key.startswith("research:"):
                    con.execute("UPDATE runs SET elapsed=elapsed+? WHERE id=?",(row["timeout"],run_id))
                con.execute("UPDATE operations SET status='unknown' WHERE run_id=? AND id=?",(run_id,key))

    def allow_retry(self, run_id, key):
        # Preserve previous outcomes in a new immutable event; next try keeps total counters.
        with self.connect() as con:
            row=con.execute("SELECT * FROM operations WHERE run_id=? AND id=?",(run_id,key)).fetchone()
            if not row or row["status"] not in ("pending","unknown","failed"):
                raise ValueError("That operation is not awaiting a deliberate retry.")
            con.execute("DELETE FROM operations WHERE run_id=? AND id=?",(run_id,key))
            con.execute("UPDATE allowances SET operation_id=operation_id||':spent' WHERE run_id=? AND kind='transient_retries' AND operation_id IN (?,?)",
                        (run_id,key,key+":transient-retry"))
            if key.endswith(":transient-retry"):
                parent=key.removesuffix(":transient-retry")
                con.execute("DELETE FROM operations WHERE run_id=? AND id=? AND status='failed'",(run_id,parent))
        self.event(run_id,"deliberate_retry",{"operation_id":key,"previous_status":row["status"],"may_repeat_charge":True})

    def event(self, run_id, kind, details):
        with self.connect() as con:
            con.execute("INSERT INTO events(run_id,at,kind,details) VALUES(?,?,?,?)",(run_id,utc_now(),kind,json.dumps(details)))

    def events(self, run_id):
        with self.connect() as con:
            rows=con.execute("SELECT at,kind,details FROM events WHERE run_id=? ORDER BY n",(run_id,)).fetchall()
        return [dict(at=r[0],kind=r[1],details=json.loads(r[2])) for r in rows]

    def claim_submission(self, run_id, event_id):
        with self.connect() as con:
            row=con.execute("SELECT status FROM submissions WHERE run_id=? AND id=?",(run_id,event_id)).fetchone()
            if row: return False
            con.execute("INSERT INTO submissions VALUES(?,?,?)",(run_id,event_id,"accepted"))
        return True
