# combined_worldcup.py
# ------------------------------------
# A fully‑featured multi‑threaded World‑Cup simulator with real-time visualizations.

from __future__ import annotations  # forward refs in type‑hints.

import math
import json
import queue
import random
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib as mpl

# Set up matplotlib for interactive mode
matplotlib.use("TkAgg")
plt.ion()

# Global style overrides for dark theme
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['text.color'] = 'white'
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'

# Color constants for visualizations
FIG_BG = "#1a3e1a"     # dark forest
AX_BG = "#276627"      # lighter forest
LINE_COLOR = 'white'
HIGHLIGHT_F = "#5cb85c"  # bright green fill
HIGHLIGHT_E = "#3e8e3e"  # dark green border

###############################################################################
#                             🔽  DATA LOADING  🔽                           #
###############################################################################

DATA_FILE = Path(__file__).with_name("team_data.json")

try:
    with DATA_FILE.open("r", encoding="utf-8") as fp:
        _team_data: Dict[str, Dict[str, Dict[str, List[str]]]] = json.load(fp)
except FileNotFoundError as err:
    raise SystemExit(
        f"❌ Required data file '{DATA_FILE}' not found.  "
        "Make sure it sits next to this script and otherwise use the extract_data.py file."
    ) from err

# These two globals are used throughout the simulator
starting_lineups: Dict[str, List[str]] = _team_data["starting_lineups"]
team_rankings: Dict[str, int] = _team_data["team_rankings"]

###############################################################################
#                               ☕  VISUALIZATIONS  ☕                        #
###############################################################################

def update_group_standings_chart(
    group_name: str,
    teams: List[Team],
    *,
    initial: bool = False,
    final: bool = False
):
    """Draw initial (zeros), live after-match, or final (highlight) bars
       with a dark theme (but keep bars in C0/C1/C2)."""
    ax = plt.gca()
    fig = plt.gcf()

    # Clear & set dark backgrounds
    ax.clear()
    fig.patch.set_facecolor("darkgreen")    # figure background
    ax.set_facecolor("green")           # plot background
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#555")

    # Title
    title_color = "#eee"
    ax.set_title(f"Group {group_name} – Standings", color=title_color)

    # Axes labels & ticks
    ax.set_xlabel("Teams", color="white", fontweight="bold")
    ax.set_ylabel("Pts / GD / GS", color="white", fontweight="bold")
    names = [t.name for t in teams]
    x = np.arange(len(names))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, color="white", fontweight="bold")
    ax.tick_params(colors="white")

    # Spine color
    for spine in ax.spines.values():
        spine.set_color("lightgreen")

    # Data bars
    if initial:
        pts = [0] * len(teams)
        gds = [0] * len(teams)
        gs = [0] * len(teams)
    else:
        pts = [t.points for t in teams]
        gds = [t.goal_difference for t in teams]
        gs = [t.goals_scored for t in teams]

    w = 0.25
    bars1 = ax.bar(x - w, pts, w, label="Pts", color="darkblue")
    bars2 = ax.bar(x, gds, w, label="GD", color="C1")
    bars3 = ax.bar(x + w, gs, w, label="GS", color="red")

    # Bar labels
    ax.bar_label(bars1, padding=3, color="white")
    ax.bar_label(bars2, padding=3, color="white")
    ax.bar_label(bars3, padding=3, color="white")

    # Highlight top2 on final
    if final:
        ranked = sorted(
            teams,
            key=lambda t: (t.points, t.goal_difference, t.goals_scored),
            reverse=True
        )
        winners = {ranked[0].name, ranked[1].name}

        for i, t in enumerate(teams):
            if t.name in winners:
                for b in (bars1, bars2, bars3):
                    b[i].set_color("gold")

        plt.pause(0.5)

    # Legend
    leg = ax.legend(frameon=False)
    for text in leg.get_texts():
        text.set_color("#ddd")

    ax.margins(x=0.02)
    fig.tight_layout(rect=[0, 0.05, 0.95, 0.95], pad=0.4)

    # Draw & pause
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.5) 

###############################################################################
#                               ☕  CAFETERIA  ☕                              #
###############################################################################

class CafeMenu:
    """Static café menu. Returns random item tuples for a given order size."""

    def __init__(self) -> None:
        # item_name ➔ (price €, prep_time ms)
        self.menu_items: Dict[str, Tuple[float, int]] = {
            "Beer": (5.0, 90),
            "Water": (3.0, 90),
            "Hot Dog": (6.0, 180),
            "Sandwich": (7.5, 240),
            "French Fries": (6.0, 300),
            "Hamburger": (10.0, 360),
        }

    def get_random_items(self, num_items: int):
        items = list(self.menu_items.items())
        return [random.choice(items) for _ in range(num_items)]


class CustomerOrder:
    """Encapsulates a spectator order flowing through the café queue."""

    def __init__(self, client_name: str, order_items, assigned_barista: str) -> None:
        self.client_name = client_name
        self.order_items = order_items
        self.total_cost = sum(item[1][0] for item in order_items)
        self.order_status = "Pending"
        self.prep_time = sum(item[1][1] for item in order_items)
        self.assigned_barista = assigned_barista
        self.order_time = None
        self.db_id = None

    def set_status(self, new_status: str):
        self.order_status = new_status
        print(
            f"Order for {self.client_name}: {new_status} (Handled by {self.assigned_barista})"
        )


class BaristaThread(threading.Thread):
    def __init__(
        self,
        barista_name: str,
        order_queue: queue.Queue,
        available_barista_queue: queue.Queue,
        db_path: Path,
        match_id: int
    ):
        super().__init__(daemon=True)
        self.barista_name = barista_name
        self.order_queue = order_queue
        self.available_barista_queue = available_barista_queue
        self.db_path = db_path
        self.match_id = match_id
        self._running = True
        self._lock = threading.Lock()

    def run(self):
        try:
            self.db_conn = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
            self.db_cursor = self.db_conn.cursor()
            self.db_cursor.execute("PRAGMA foreign_keys = ON")

            while self._running or not self.order_queue.empty():
                try:
                    order: CustomerOrder = self.order_queue.get(timeout=1)
                except queue.Empty:
                    continue

                with self._lock:
                    order.set_status("In Progress")
                    time.sleep(order.prep_time / 1000)
                    order.set_status("Completed")

                    print(
                        f"{order.client_name}, your order is ready! Total: €{order.total_cost:.2f} "
                        f"(Handled by {self.barista_name})\n"
                    )

                    completion = datetime.now()
                    actual_ms = int((completion - order.order_time).total_seconds() * 1000)

                    if getattr(order, "db_id", None) is None:
                        print(f"⚠️ Skipping DB update: order for {order.client_name} has no db_id.")
                        self.available_barista_queue.put(self.barista_name)
                        self.order_queue.task_done()
                        continue

                    try:
                        self.db_cursor.execute(
                            """
                            UPDATE orders
                            SET actual_prep_ms = ?, completion_time = ?
                            WHERE id = ?
                            """,
                            (actual_ms, completion, order.db_id)
                        )
                        self.db_conn.commit()
                    except sqlite3.Error as e:
                        print(f"Database error: {e}")
                        continue

                    self.order_queue.task_done()
                    self.available_barista_queue.put(self.barista_name)
        except Exception as e:
            print(f"Error in barista thread {self.barista_name}: {e}")
        finally:
            if hasattr(self, 'db_conn'):
                self.db_conn.close()

    def stop(self):
        self._running = False


###############################################################################
#                              🏟️  MATCH MODEL  🏟️                           #
###############################################################################

class Team:
    """Lightweight container for aggregate group‑stage statistics."""

    def __init__(self, name: str, ranking: int, lineup: List[str]):
        self.name = name
        self.ranking = ranking
        self.lineup = lineup
        self.points = 0
        self.goals_scored = 0
        self.goals_conceded = 0
        self.goal_difference = 0

    def update_stats(self, goals_for: int, goals_against: int):
        """Update group‑table stats after a match."""
        self.goals_scored += goals_for
        self.goals_conceded += goals_against
        self.goal_difference = self.goals_scored - self.goals_conceded
        if goals_for > goals_against:
            self.points += 3
        elif goals_for == goals_against:
            self.points += 1


class Stadium:
    """Represents a match venue with bathrooms guarded by semaphores."""

    def __init__(self, name: str, location: str, capacity: int):
        self.name = name
        self.location = location
        self.capacity = capacity
        self.lock = threading.Lock()
        self.bathrooms: Dict[str, threading.Semaphore] = {
            "North Bathroom": threading.Semaphore(4),
            "South Bathroom": threading.Semaphore(4),
            "East Bathroom": threading.Semaphore(4),
            "West Bathroom": threading.Semaphore(4),
            "VIP Lounge Bathroom": threading.Semaphore(2),
        }


# Global lock for match commentary
match_lock = threading.Lock()


class Spectator:
    """A stadium guest that can: enter, visit bathroom, and order food."""

    def __init__(self, spectator_id: int, match: "Match"):
        self.spectator_id = spectator_id
        self.match = match
        self.gate_number = random.randint(1, 10)

    def enter_stadium(self):
        print(
            f"Spectator {self.spectator_id} entered {self.match.stadium.name} "
            f"through Gate {self.gate_number}"
        )
        time.sleep(0.2)

    def use_bathroom(self):
        bathroom_name, stall_sem = random.choice(list(self.match.stadium.bathrooms.items()))

        print(f"Spectator {self.spectator_id} is waiting for {bathroom_name}")
        t0 = time.perf_counter()
        with stall_sem:
            wait = time.perf_counter() - t0
            print(f"Spectator {self.spectator_id} entered {bathroom_name}")
            t1 = time.perf_counter()
            time.sleep(0.5)
            duration = time.perf_counter() - t1
            print(f"Spectator {self.spectator_id} left {bathroom_name}")

            try:
                conn, cursor = self.match._get_db_connection()
                with self.match._db_lock:
                    cursor.execute(
                        """
                        INSERT INTO bathroom_usage(
                          match_id, spectator_id, bathroom_name, wait_time, use_duration
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (self.match.match_id, self.spectator_id, bathroom_name, wait, duration)
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error during bathroom use: {e}")

    def order_food(
        self,
        menu: CafeMenu,
        order_queue: queue.Queue,
        available_barista_queue: queue.Queue,
        sequence_lock: threading.Lock,
    ):
        """Places an order and enqueues it for preparation."""
        assigned_barista = available_barista_queue.get()
        with sequence_lock:
            num_items = random.randint(1, 3)
            selected_items = menu.get_random_items(num_items)
            items_str = ", ".join(f"{item[0]} (€{item[1][0]:.2f})" for item in selected_items)
            print(
                f"Spectator {self.spectator_id} ordered: {items_str} "
                f"(At {assigned_barista})"
            )
            new_order = CustomerOrder(
                f"Spectator-{self.spectator_id}", selected_items, assigned_barista
            )
            print(
                f"Spectator {self.spectator_id}, your order has been received! "
                f"Estimated preparation time: {new_order.prep_time} ms\n"
            )

        with sequence_lock:
            try:
                conn, cursor = self.match._get_db_connection()
                with self.match._db_lock:
                    order_time = datetime.now()
                    new_order.order_time = order_time
                    items_json = json.dumps([item[0] for item in selected_items])
                    cursor.execute(
                        """
                        INSERT INTO orders(
                        match_id, spectator_id, order_time,
                        items_json, total_cost, estimated_prep_ms
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.match.match_id,
                            self.spectator_id,
                            order_time,
                            items_json,
                            new_order.total_cost,
                            new_order.prep_time
                        )
                    )
                    conn.commit()
                    new_order.db_id = cursor.lastrowid

                order_queue.put(new_order)
            except Exception as e:
                print(f"Error processing order: {e}")
                available_barista_queue.put(assigned_barista)


class Match(threading.Thread):
    def __init__(
        self,
        team1: Team,
        team2: Team,
        stadium: Stadium,
        match_time: datetime,
        db_path: Path,
        team_id_map: Dict[str, int],
        knockout: bool = False,
    ):
        super().__init__(daemon=True)

        self.db_path = db_path
        self.team_id_map = team_id_map

        # Thread-safe collections and locks
        self._spectator_list_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._event_lock = threading.Lock()
        
        # Database connection pool
        self._db_connections = {}
        self._db_cursors = {}

        self.team1 = team1
        self.team2 = team2
        self.stadium = stadium
        self.match_time = match_time
        self.knockout = knockout

        # In‑stadium state
        self.spectator_count = random.randint(5, 10)
        self.score1 = 0
        self.score2 = 0
        self.winner: Team | None = None

        # Thread‑coordination primitives
        self.match_over_event = threading.Event()
        self.spectator_entry_event = threading.Event()
        self.cafeteria_open_event = threading.Event()
        self.spectator_lock = threading.Lock()

        self.next_spectator_id = 1
        self.entered_spectators: List[Spectator] = []

        self.match_id: int = -1

    def _get_db_connection(self):
        """Get a thread-specific database connection"""
        thread_id = threading.get_ident()
        if thread_id not in self._db_connections:
            with self._db_lock:
                if thread_id not in self._db_connections:
                    conn = sqlite3.connect(
                        str(self.db_path),
                        detect_types=sqlite3.PARSE_DECLTYPES,
                        check_same_thread=False
                    )
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA foreign_keys = ON")
                    self._db_connections[thread_id] = conn
                    self._db_cursors[thread_id] = cursor
        return self._db_connections[thread_id], self._db_cursors[thread_id]

    def _close_db_connection(self):
        """Close the thread-specific database connection"""
        thread_id = threading.get_ident()
        if thread_id in self._db_connections:
            with self._db_lock:
                if thread_id in self._db_connections:
                    self._db_connections[thread_id].close()
                    del self._db_connections[thread_id]
                    del self._db_cursors[thread_id]

    def simulate_spectator_entry(self, count: int):
        """Spawn *count* new Spectator objects and mark them as present."""
        with self._spectator_list_lock:
            new_spectators = [
                Spectator(self.next_spectator_id + i, self) for i in range(count)
            ]
            self.next_spectator_id += count
            self.entered_spectators.extend(new_spectators)
            
        for s in new_spectators:
            s.enter_stadium()
            try:
                conn, cursor = self._get_db_connection()
                with self._db_lock:
                    entry_time = datetime.now()
                    cursor.execute(
                        "INSERT INTO spectator_entries(match_id, spectator_id, entry_time) VALUES (?, ?, ?)",
                        (self.match_id, s.spectator_id, entry_time)
                    )
                    conn.commit()
            except sqlite3.Error as e:
                print(f"Database error during spectator entry: {e}")

    def continuous_spectator_behavior(self):
        """Background thread: trickle of new arrivals + random bathroom trips."""
        try:
            while not self.match_over_event.is_set():
                if not self.spectator_entry_event.is_set():
                    self.simulate_spectator_entry(random.randint(1, 3))

                with self._spectator_list_lock:
                    if self.entered_spectators:
                        spectator = random.choice(self.entered_spectators)
                        try:
                            spectator.use_bathroom()
                        except Exception as e:
                            print(f"Error during bathroom use: {e}")

                time.sleep(random.uniform(1, 2))
        except Exception as e:
            print(f"Error in spectator behavior thread: {e}")
        finally:
            self._close_db_connection()

    def continuous_cafeteria_service(self):
        """Background thread: keeps three baristas alive for the whole match."""
        menu_board = CafeMenu()
        order_queue: queue.Queue = queue.Queue()
        sequence_lock = threading.Lock()
        available_barista_queue: queue.Queue = queue.Queue()

        baristas: List[BaristaThread] = []
        try:
            for name in [f"Stand-{i + 1}" for i in range(3)]:
                available_barista_queue.put(name)
                b = BaristaThread(
                    name,
                    order_queue,
                    available_barista_queue,
                    self.db_path,
                    self.match_id
                )
                b.start()
                baristas.append(b)

            while not self.match_over_event.is_set():
                if self.cafeteria_open_event.is_set():
                    with self._spectator_list_lock:
                        if not self.entered_spectators:
                            time.sleep(1)
                            continue
                        customers = random.sample(
                            self.entered_spectators,
                            min(3, len(self.entered_spectators))
                        )
                    
                    for spectator in customers:
                        try:
                            spectator.order_food(
                                menu_board,
                                order_queue,
                                available_barista_queue,
                                sequence_lock
                            )
                        except Exception as e:
                            print(f"Error processing order for spectator {spectator.spectator_id}: {e}")
                            continue
                    time.sleep(random.uniform(3, 5))
                else:
                    time.sleep(1)

        except Exception as e:
            print(f"Error in cafeteria service: {e}")
        finally:
            try:
                order_queue.join()
                for b in baristas:
                    b.stop()
                for b in baristas:
                    if b.is_alive():
                        b.join(timeout=2)
                print("Cafeteria closed after match!\n")
            except Exception as e:
                print(f"Error during cafeteria shutdown: {e}")
            
            self._close_db_connection()

    def simulate_half_events(self, half: str):
        """Produce randomised commentary & scoring chances for one half."""
        print(f"\n\n--- {half} begins ---")
        half_start_time = time.time()
        events = ["shot", "corner", "tackle", "foul", "dribble", "pass"]
        
        try:
            conn, cursor = self._get_db_connection()
            for _ in range(random.randint(10, 15)):
                time.sleep(random.uniform(0.5, 1.2))

                minute = int((time.time() - half_start_time) // 1)
                acting_team = random.choices(
                    [self.team1, self.team2],
                    weights=[self.team1.ranking, self.team2.ranking]
                )[0]
                opponent = self.team2 if acting_team is self.team1 else self.team1
                event = random.choices(events, weights=[40, 15, 15, 10, 10, 10])[0]

                shooter = None
                if event == "shot":
                    shooter = random.choice(acting_team.lineup[1:11])
                    bias = (acting_team.ranking / (acting_team.ranking + opponent.ranking)) - 0.5
                    p_goal = 0.22 + 0.4 * bias
                    is_goal = (random.random() < p_goal)
                else:
                    is_goal = False

                with self._db_lock:
                    acting_id = self.team_id_map[acting_team.name]
                    cursor.execute(
                        """
                        INSERT INTO match_events(
                          match_id, event_minute, event_type,
                          acting_team_id, player_name, success
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (self.match_id, minute, event, acting_id, shooter or "", is_goal)
                    )
                    conn.commit()

                if event == "shot":
                    if is_goal:
                        if acting_team is self.team1:
                            self.score1 += 1
                        else:
                            self.score2 += 1
                        print(
                            f"GOAL! {shooter} from {acting_team.name} scored! 🎉\n"
                            f"Score: {self.team1.name} {self.score1} – {self.score2} {self.team2.name}"
                        )
                    else:
                        print(f"{shooter} from {acting_team.name} took a shot but missed.")
                elif event == "corner":
                    print(f"{acting_team.name} won a corner kick.")
                elif event == "tackle":
                    print(
                        f"{random.choice(acting_team.lineup)} tackled "
                        f"{random.choice(opponent.lineup)}."
                    )
                elif event == "foul":
                    print(f"{random.choice(acting_team.lineup)} committed a foul.")
                elif event == "dribble":
                    print(f"{random.choice(acting_team.lineup)} made a skillful dribble.")
                else:  # pass
                    print(f"{random.choice(acting_team.lineup)} completed a nice pass.")
        except Exception as e:
            print(f"Error during match simulation: {e}")

    def run(self):
        """Entry‑point of the Match thread."""
        try:
            with match_lock:
                print(f"\nMatch Scheduled: {self.team1.name} vs {self.team2.name} ...")
                print("☕ Cafeteria ready for pre‑match rush!")

                # First wave of spectators
                self.simulate_spectator_entry(self.spectator_count)

                # Start background threads
                spectator_thread = threading.Thread(
                    target=self.continuous_spectator_behavior,
                    daemon=True
                )
                cafeteria_thread = threading.Thread(
                    target=self.continuous_cafeteria_service,
                    daemon=True
                )
                
                spectator_thread.start()
                cafeteria_thread.start()

                # Pre-match café rush
                self.cafeteria_open_event.set()
                time.sleep(random.uniform(5, 10))
                self.cafeteria_open_event.clear()

                # First Half
                print("\n=== FIRST HALF BEGINS ===")
                self.cafeteria_open_event.set()
                self.simulate_half_events("First Half")
                self.cafeteria_open_event.clear()

                # Half Time
                print("\n--- Half Time ---")
                print(
                    f"Half Time Score: {self.team1.name} {self.score1} – {self.score2} {self.team2.name}"
                )
                self.cafeteria_open_event.set()
                print("☕ Cafeteria ready for half‑time rush!")
                time.sleep(random.uniform(5, 10))
                self.cafeteria_open_event.clear()
                time.sleep(2)

                # Second Half
                print("\n=== SECOND HALF BEGINS ===")
                self.cafeteria_open_event.set()
                self.simulate_half_events("Second Half")
                self.cafeteria_open_event.clear()

                # Match End
                self.match_over_event.set()
                
                # Wait for background threads
                spectator_thread.join(timeout=5)
                cafeteria_thread.join(timeout=5)

                print(
                    f"\nMatch Finished: {self.team1.name} {self.score1} – {self.score2} {self.team2.name}"
                )

                try:
                    conn, cursor = self._get_db_connection()
                    with self._db_lock:
                        cursor.execute(
                            "UPDATE matches SET score1 = ?, score2 = ? WHERE id = ?",
                            (self.score1, self.score2, self.match_id)
                        )
                        conn.commit()
                except sqlite3.Error as e:
                    print(f"Database error updating match score: {e}")

                if self.knockout:
                    self._resolve_knockout()
                else:
                    self.team1.update_stats(self.score1, self.score2)
                    self.team2.update_stats(self.score2, self.score1)

        except Exception as e:
            print(f"Error in match thread: {e}")
        finally:
            self._close_db_connection()

    def _resolve_knockout(self):
        if self.score1 != self.score2:
            self.winner = self.team1 if self.score1 > self.score2 else self.team2
            return

        print("\nThe match is tied! Proceeding to Extra Time...")
        self.simulate_half_events("Extra Time")
        if self.score1 != self.score2:
            self.winner = self.team1 if self.score1 > self.score2 else self.team2
            return

        print("\nStill tied after Extra Time! Penalty Shootout begins!")
        self.winner = self._simulate_penalty_shootout()
        print(f"Penalty Shootout Winner: {self.winner.name}")

    def _simulate_penalty_shootout(self) -> Team:
        score1 = score2 = 0
        for i in range(5):
            if random.random() < 0.75:
                score1 += 1
            if random.random() < 0.75:
                score2 += 1
            print(f"Penalty {i+1}: {self.team1.name} {score1} – {score2} {self.team2.name}")

        round_num = 6  # sudden death
        while score1 == score2:
            print(f"Sudden Death Round {round_num}!")
            if random.random() < 0.75:
                score1 += 1
            if random.random() < 0.75:
                score2 += 1
            print(f"{self.team1.name} {score1} – {score2} {self.team2.name}")
            round_num += 1

        return self.team1 if score1 > score2 else self.team2 

###############################################################################
#                           🏆  TOURNAMENT MODEL  🏆                          #
###############################################################################

class WorldCup:
    """Sets up groups, schedules all matches and advances winners with DB logging."""

    def __init__(self, teams: List[str], stadiums: List[str]):
        # ─── Prep interactive mode & bracket figure ───
        plt.ion()
        self.master_fig, self.master_ax = plt.subplots(figsize=(8, 12), dpi=100)
        self.master_fig.canvas.manager.set_window_title("World Cup")
        self.master_fig.tight_layout(rect=[0, 0, 1, 0.93])

        self._round_pairs: Dict[int, List[Tuple[Team, Team]]] = {}
        self._team_name_texts: Dict[Tuple[int, int, int], matplotlib.text.Text] = {}
        self._positions: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # 1) IN-MEMORY SETUP
        self.teams = [Team(name, team_rankings[name], starting_lineups[name])
                      for name in teams]
        self.stadiums = [Stadium(name, "Location", random.randint(40_000, 90_000))
                         for name in stadiums]
        # For 2026: 12 groups  of 4 teams each → 12×4 = 48 slots
        self.group_size = 4
        self.groups: Dict[str, List[Team]] = {chr(65 + i): [] for i in range(12)}  # A…P

        self.start_date = datetime(2026, 6, 1, 12, 0)
        self.knockout_teams: List[Team] = []

        # 2) DATABASE SETUP (project-relative path)
        # 1️⃣ keep the path for handing out to each thread
        self.db_path = Path(__file__).parent / "worldcup.db"

        # main-thread connection (only used for seeding & schedule logging)
        self.conn = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON")

        # 3) SCHEMA CREATION
        self._create_schema()

        # 4) SEED TEAMS IN DB
        self._seed_teams()

    def _create_schema(self):
        """Create all tables if they don't exist"""
        self.cursor.executescript("""
        PRAGMA foreign_keys = ON;

        DROP TABLE IF EXISTS teams;  -- Drop and recreate to ensure clean schema
        CREATE TABLE teams (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT UNIQUE,
            ranking  INTEGER
        );

        DROP TABLE IF EXISTS matches;
        CREATE TABLE matches (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            team1_id       INTEGER,
            team2_id       INTEGER,
            match_datetime TIMESTAMP,
            stadium        TEXT,
            score1         INTEGER,
            score2         INTEGER,
            knockout       BOOLEAN,
            FOREIGN KEY(team1_id) REFERENCES teams(id),
            FOREIGN KEY(team2_id) REFERENCES teams(id)
        );

        DROP TABLE IF EXISTS match_events;
        CREATE TABLE match_events (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id       INTEGER,
            event_minute   INTEGER,
            event_type     TEXT,
            acting_team_id INTEGER,
            player_name    TEXT,
            success        BOOLEAN,
            FOREIGN KEY(match_id) REFERENCES matches(id),
            FOREIGN KEY(acting_team_id) REFERENCES teams(id)
        );

        DROP TABLE IF EXISTS spectator_entries;
        CREATE TABLE spectator_entries (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id       INTEGER,
            spectator_id   INTEGER,
            entry_time     TIMESTAMP,
            FOREIGN KEY(match_id) REFERENCES matches(id)
        );

        DROP TABLE IF EXISTS bathroom_usage;
        CREATE TABLE bathroom_usage (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id       INTEGER,
            spectator_id   INTEGER,
            bathroom_name  TEXT,
            wait_time      REAL,
            use_duration   REAL,
            FOREIGN KEY(match_id) REFERENCES matches(id)
        );

        DROP TABLE IF EXISTS orders;
        CREATE TABLE orders (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id           INTEGER,
            spectator_id       INTEGER,
            order_time         TIMESTAMP,
            items_json         TEXT,
            total_cost         REAL,
            estimated_prep_ms  INTEGER,
            actual_prep_ms     INTEGER,
            completion_time    TIMESTAMP,
            FOREIGN KEY(match_id) REFERENCES matches(id)
        );
        """)
        self.conn.commit()

    def _seed_teams(self):
        """Insert each team into DB and store its ID."""
        self.team_id_map: Dict[str, int] = {}
        
        # First, clear any existing teams to avoid duplicates
        self.cursor.execute("DELETE FROM teams")
        self.conn.commit()
        
        # Then insert all teams
        for team in self.teams:
            try:
                self.cursor.execute(
                    "INSERT INTO teams(name, ranking) VALUES (?, ?)",
                    (team.name, team.ranking)
                )
                team_id = self.cursor.lastrowid
                self.team_id_map[team.name] = team_id
            except sqlite3.Error as e:
                print(f"Error inserting team {team.name}: {e}")
                raise
        
        self.conn.commit()
        
        # Verify all teams were inserted
        self.cursor.execute("SELECT COUNT(*) FROM teams")
        count = self.cursor.fetchone()[0]
        if count != len(self.teams):
            raise RuntimeError(f"Expected {len(self.teams)} teams in database, but found {count}")

    def assign_teams_to_groups(self):
        """Round-robin distribution (3 teams per group)."""
        group_keys = list(self.groups.keys())
        team_index = 0
        while team_index < len(self.teams):
            for group in group_keys:
                if len(self.groups[group]) < self.group_size and team_index < len(self.teams):
                    self.groups[group].append(self.teams[team_index])
                    team_index += 1
                if team_index >= len(self.teams):
                    break

    def schedule_group_stage(self):
        current_date = self.start_date
        matches_per_day = 0

        for group_label, group_teams in self.groups.items():
            plt.figure(self.master_fig.number)
            ax = self.master_ax
            ax.clear()
            self.master_fig.patch.set_facecolor(FIG_BG)
            ax.set_facecolor(AX_BG)
            self.master_fig.canvas.manager.set_window_title(f"Group {group_label}")

            # 3) now draw the initial standings
            update_group_standings_chart(group_label, group_teams, initial=True)

            print(f"\n--- Playing Group {group_label} Matches ---")
            for i in range(len(group_teams)):
                for j in range(i + 1, len(group_teams)):
                    if matches_per_day >= 6:
                        current_date += timedelta(days=1)
                        matches_per_day = 0

                    match_time = current_date.replace(
                        hour=[13, 16, 19, 21][matches_per_day % 4]
                    )
                    stadium = random.choice(self.stadiums)

                    # Log the match in DB
                    self.cursor.execute(
                        """
                        INSERT INTO matches(
                          team1_id, team2_id, match_datetime,
                          stadium, score1, score2, knockout
                        ) VALUES (?, ?, ?, ?, 0, 0, ?)
                        """,
                        (
                            self.team_id_map[group_teams[i].name],
                            self.team_id_map[group_teams[j].name],
                            match_time,
                            stadium.name,
                            False
                        )
                    )
                    match_id = self.cursor.lastrowid
                    self.conn.commit()

                    # Run the match
                    match = Match(
                        group_teams[i],
                        group_teams[j],
                        stadium,
                        match_time,
                        self.db_path,
                        self.team_id_map,
                        False
                    )
                    match.match_id = match_id
                    match.start()
                    match.join()

                    # Update the live chart
                    update_group_standings_chart(group_label, group_teams)

                    matches_per_day += 1

            # Final standings for this group
            update_group_standings_chart(group_label, group_teams, final=True)
            plt.pause(2.0)
            ax.clear()

        print("\nGroup Stage Completed!")
        self.display_standings()
        self._prepare_knockout_stage()

    def display_standings(self):
        print("\nGroup Stage Standings:")
        for group, teams in self.groups.items():
            sorted_teams = sorted(
                teams, key=lambda x: (-x.points, -x.goal_difference, -x.goals_scored)
            )
            print(f"\nGroup {group}:")
            print("Team | Pts | GS | GC | GD")
            for t in sorted_teams:
                print(
                    f"{t.name} | {t.points} | {t.goals_scored} | {t.goals_conceded} | {t.goal_difference}"
                )

    def _prepare_knockout_stage(self):
        print("\nPreparing Knockout Stage...")
        third_place_candidates: List[Team] = []
        for teams in self.groups.values():
            ranked = sorted(teams, key=lambda x: (-x.points, -x.goal_difference, -x.goals_scored))
            self.knockout_teams.extend(ranked[:2])  # top two auto‑advance
            third_place_candidates.append(ranked[2])
        self.knockout_teams.extend(
            sorted(third_place_candidates, key=lambda x: (-x.points, -x.goal_difference, -x.goals_scored))[:8]
        )
        print(
            "\nQualified Teams: ", [t.name for t in self.knockout_teams]
        )
        self._print_knockout_probabilities()
        self._simulate_knockout_stage()

    def _print_knockout_probabilities(self):
        total_rating = sum(t.ranking for t in self.knockout_teams)
        print("\nEstimated win probabilities:")
        for t in self.knockout_teams:
            print(f"{t.name}: {t.ranking / total_rating * 100:.1f}%")

    def _simulate_knockout_stage(self):
        # 0) prep bracket layout for whatever # teams we have (now 32)
        self._init_bracket()
        teams = list(self.knockout_teams)
        # map round‐index → label (now 5 rounds: 32→16→8→4→2)
        round_labels = {
            0: "Round of 32",
            1: "Round of 16",
            2: "Quarter Finals",
            3: "Semi Finals",
            4: "Final"
        }
        round_number = 0

        # scheduling template
        base_date = datetime(2026, 7, 1)
        time_slots = [13, 16, 19, 21]
        slot_index = 0
        day_offset = 0

        # seed the first round's pairs
        self._round_pairs[0] = [
            (teams[2 * i], teams[2 * i + 1])
            for i in range(len(teams) // 2)
        ]

        # loop until we have a champion
        while len(teams) > 1:
            label = round_labels[round_number]
            print(f"\n--- {label} ---")

            # update the suptitle
            self.master_fig.suptitle(
                f"Knockout: {label}",
                fontsize=28, fontweight="bold",
                color="white", backgroundcolor=AX_BG, y=0.90
            )
            self.master_fig.canvas.draw()
            self.master_fig.canvas.flush_events()
            plt.pause(0.2)

            next_round = []
            # run each match in this round
            for m_idx in range(len(teams) // 2):
                a, b = teams[2 * m_idx], teams[2 * m_idx + 1]

                # schedule datetime
                rnd_stadium = random.choice(self.stadiums)
                hour = time_slots[slot_index % len(time_slots)]
                mtime = (base_date + timedelta(days=day_offset)).replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )
                slot_index += 1
                if slot_index % len(time_slots) == 0:
                    day_offset += 1

                # log & highlight
                self.cursor.execute(
                    "INSERT INTO matches(team1_id,team2_id,match_datetime,stadium,score1,score2,knockout) "
                    "VALUES(?,?,?,?,0,0,1)",
                    (self.team_id_map[a.name], self.team_id_map[b.name], mtime, rnd_stadium.name)
                )
                match_id = self.cursor.lastrowid
                self.conn.commit()

                self._highlight_match(round_number, m_idx)
                self.master_fig.canvas.draw()
                self.master_fig.canvas.flush_events()
                plt.pause(0.5)

                # play!
                match = Match(a, b, rnd_stadium, mtime, self.db_path, self.team_id_map, True)
                match.match_id = match_id
                match.start()
                match.join()

                # record result + advance
                winner = match.winner
                loser = b if winner is a else a
                self._resolve_and_advance(
                    round_number,
                    m_idx,
                    winner,
                    loser,
                    match.score1,
                    match.score2
                )

                self.master_fig.canvas.draw()
                self.master_fig.canvas.flush_events()
                plt.pause(1.0)

                next_round.append(winner)

            # prepare the next‐round pairing map
            self._round_pairs[round_number + 1] = [
                (next_round[2 * i], next_round[2 * i + 1])
                for i in range(len(next_round) // 2)
            ]

            # move on
            teams = next_round
            round_number += 1

        # champion
        print(f"\n🏆 The World Cup Champion is: {teams[0].name} 🏆")
        # … your celebration overlay follows …

        ## ─── Celebration Overlay ───
        # 1) Remove any existing figure suptitle
        if getattr(self.master_fig, "_suptitle", None):
            self.master_fig._suptitle.remove()
        # Clear out the old bracket
        self.master_ax.clear()
        self.master_ax.set_axis_off()

        # Make the figure a bit wider for the banner
        self.master_fig.set_size_inches(10, 5)
        self.master_fig.patch.set_facecolor(FIG_BG)

        # Draw the main title
        self.master_ax.text(
            0.5, 0.65,
            "WORLD CUP CHAMPIONS 2026",
            ha="center", va="center",
            fontsize=36,
            fontweight="bold",
            color="gold",
            transform=self.master_ax.transAxes
        )

        # Draw the winning team's name
        self.master_ax.text(
            0.5, 0.40,
            teams[0].name,
            ha="center", va="center",
            fontsize=28,
            fontweight="bold",
            color="white",
            transform=self.master_ax.transAxes
        )

        # 6) Manually adjust margins so nothing gets clipped
        self.master_fig.subplots_adjust(
            left=0.1,  # a little space on the left
            right=0.9,  # a little space on the right
            top=0.8,  # keep a bit above the top text
            bottom=0.1  # keep a bit below the bottom text
        )

        # 7) Render and pause so the user gets to see it
        self.master_fig.canvas.draw()
        self.master_fig.canvas.flush_events()
        plt.pause(5.0)

    def _init_bracket(self):
        """Only do all of this on the main thread"""
        if threading.current_thread() is not threading.main_thread():
            return

        # 1) clear & prep
        plt.figure(self.master_fig.number)
        ax = self.master_ax
        ax.clear()
        ax.set_axis_off()
        self.master_fig.patch.set_facecolor(FIG_BG)
        ax.set_facecolor(AX_BG)
        # ── SHIFT ENTIRE BRACKET LEFT BY 5 CM ──
        fig = self.master_fig
        width_inch = fig.get_size_inches()[0]
        shift_frac = (1.0 / 2.54) / width_inch  # 5 cm → inches → fraction of figure width
        pos = ax.get_position()  # current [x0, y0, width, height]
        ax.set_position([pos.x0 + shift_frac, pos.y0, pos.width, pos.height])

        # 2) compute sizes
        n_teams = len(self.knockout_teams)
        n_rounds = int(math.log2(n_teams))
        # slight push right so names fit:
        left_edge, right_edge = 0.05, 0.8
        x_centers = np.linspace(left_edge, right_edge, n_rounds)
        # **no** extra 1.1 factor here
        base_block = 1.0 / (n_teams - 1)
        dx = (x_centers[1] - x_centers[0]) * 0.3
        name_offset = dx * 0.00001
        self._dx = dx

        # 3) title
        self.master_fig.suptitle(
            f"Knockout: Round of {n_teams}",
            fontsize=18, fontweight="bold", color="white",
            backgroundcolor=AX_BG, y=0.93
        )

        LINE_KW = dict(color=LINE_COLOR, linewidth=2,
                       solid_capstyle='round', alpha=0.9)
        self._positions = {}

        # 4) draw each round
        for r, x_mid in enumerate(x_centers):
            n_matches = n_teams // (2 ** (r + 1))
            block_h = base_block * (2 ** r)

            for m in range(n_matches):
                if r == 0:
                    y_top = 1.0 - (2 * m) * block_h
                    y_bot = y_top - block_h
                else:
                    y_top = self._positions[(r - 1, 2 * m, 0)][1]
                    y_bot = self._positions[(r - 1, 2 * m + 1, 1)][1]

                x0 = x_mid - dx
                # stash coords
                self._positions[(r, m, 0)] = (x0, y_top)
                self._positions[(r, m, 1)] = (x0, y_bot)

                # bracket lines
                ax.add_line(Line2D([x0, x_mid], [y_top, y_top], **LINE_KW))
                ax.add_line(Line2D([x0, x_mid], [y_bot, y_bot], **LINE_KW))
                ax.add_line(Line2D([x_mid, x_mid], [y_top, y_bot], **LINE_KW))

                # seed R32 names
                if r == 0:
                    t1 = self.knockout_teams[2 * m]
                    t2 = self.knockout_teams[2 * m + 1]

                    # compute text-x once using our offset
                    x_text = x0 - name_offset

                    # draw and capture the Text objects
                    t1_text = ax.text(
                        x_text, y_top, t1.name,
                        ha="right", va="center",
                        fontsize=10, fontweight="bold", color="white"
                    )
                    t2_text = ax.text(
                        x_text, y_bot, t2.name,
                        ha="right", va="center",
                        fontsize=10, fontweight="bold", color="white"
                    )

                    # stash them so we can center our "✕" later
                    self._team_name_texts[(0, m, 0)] = t1_text
                    self._team_name_texts[(0, m, 1)] = t2_text

        # 5) prepare highlights & winner slots
        self._highlight_boxes = {}
        self._winner_texts = {}
        for r, x_mid in enumerate(x_centers):
            n_matches = n_teams // (2 ** (r + 1))
            for m in range(n_matches):
                x0, y_top = self._positions[(r, m, 0)]
                _, y_bot = self._positions[(r, m, 1)]

                box = Rectangle((x0 - 0.01, y_bot - 0.01),
                                width=dx + 0.02,
                                height=(y_top - y_bot) + 0.02,
                                facecolor=HIGHLIGHT_F, alpha=0.15,
                                visible=False)
                ax.add_patch(box)
                self._highlight_boxes[(r, m)] = box

                nx = x_mid + dx - 0.03
                ny = (y_top + y_bot) / 2
                txt = ax.text(nx, ny, "",
                              ha="left", va="center",
                              fontsize=10, fontweight="bold", color="white")
                self._winner_texts[(r, m)] = txt

        # 6) final layout
        ax.set_xlim(left_edge - 0.02, right_edge + 0.02)
        ax.set_ylim(0, 1)
        self.master_fig.tight_layout(pad=4.0)
        plt.pause(0.1)

    def _highlight_match(self, round_idx: int, match_idx: int):
        ax = self.master_ax

        # 1) clear any previous highlight
        for p in getattr(self, "_hl_patches", []):
            p.remove()
        self._hl_patches = []

        # 2) endpoints of the two bracket lines
        x0, y_top = self._positions[(round_idx, match_idx, 0)]
        _, y_bot = self._positions[(round_idx, match_idx, 1)]

        # 3) recompute horizontal span (same logic you used in init)
        n_teams = len(self.knockout_teams)
        n_rounds = int(math.log2(n_teams))
        dx = (1.0 / n_rounds) * 0.3

        # 4) vertical padding
        pad_v = 0.02
        bottom = y_bot - pad_v
        height = (y_top - y_bot) + 2 * pad_v

        # 5) for Round-of-32, cover the text objects rather than fixed x0
        if round_idx == 0:
            t1 = self._team_name_texts[(0, match_idx, 0)]
            t2 = self._team_name_texts[(0, match_idx, 1)]
            x1, _ = t1.get_position()
            x2, _ = t2.get_position()
            left = min(x1, x2) - 0.01  # a bit extra left of the leftmost name
            right = x0 + dx + 0.01  # a bit extra right of the bracket line
        else:
            # other rounds can stay anchored at x0
            left = x0 - 0.01
            right = x0 + dx + 0.01

        width = right - left

        # 6) draw the box
        box = Rectangle(
            (left, bottom),
            width,
            height,
            transform=ax.transData,
            facecolor=HIGHLIGHT_F,
            edgecolor=HIGHLIGHT_E,
            linewidth=2,
            alpha=0.7,
            capstyle='round'
        )
        ax.add_patch(box)
        self._hl_patches.append(box)

    def _resolve_and_advance(
            self,
            round_idx: int,
            match_idx: int,
            winner: Team,
            loser: Team,
            score1: int,
            score2: int
    ):
        ax = self.master_ax

        # Figure out which side (0=top,1=bottom) the loser was on
        a, b = self._round_pairs[round_idx][match_idx]
        side = 0 if loser is a else 1

        # 1) Locate the Text object for that loser's name
        if (round_idx, match_idx, side) in self._team_name_texts:
            txt = self._team_name_texts[(round_idx, match_idx, side)]
        elif round_idx > 0:
            # in later rounds, loser was drawn as a previous-round winner
            child_r = round_idx - 1
            child_m = 2 * match_idx + side
            txt = self._winner_texts.get((child_r, child_m))
        else:
            txt = None

        # 2) Compute its exact center (data coords) for the "✕"
        if txt:
            # draw to ensure our renderer is up-to-date
            self.master_fig.canvas.draw()
            renderer = self.master_fig.canvas.get_renderer()
            # get the text's bounding-box in display coords
            bbox = txt.get_window_extent(renderer=renderer)
            disp_x = bbox.x0 + bbox.width / 2
            disp_y = bbox.y0 + bbox.height / 2
            # invert back to data coords
            inv = ax.transData.inverted()
            x_center, y_center = inv.transform((disp_x, disp_y))
        else:
            # fallback if for some reason txt wasn't found
            x_center, y_center = self._positions[(round_idx, match_idx, side)]

        # finally draw the big red "✕":
        ax.text(
            x_center, y_center, "✕",
            color="red", fontsize=20, fontweight="bold",
            ha="center", va="center"
        )

        # 4) Also draw the numeric score at the midpoint of the two lines
        x0, y_top = self._positions[(round_idx, match_idx, 0)]
        _, y_bot = self._positions[(round_idx, match_idx, 1)]
        x_score = x0 + self._dx * 0.5
        y_score = (y_top + y_bot) / 2
        ax.text(
            x_score, y_score,
            f"{score1} – {score2}",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(
                boxstyle="round,pad=0.1",
                facecolor=AX_BG,
                edgecolor=LINE_COLOR,
                alpha=0.8
            )
        )

        # 5) Finally, write the winner into the next-round slot
        self._winner_texts[(round_idx, match_idx)].set_text(winner.name)

    def start_tournament(self):
        """User‑facing entry‑point."""
        self.assign_teams_to_groups()
        print("Teams assigned to groups:")
        for group, teams in self.groups.items():
            print(f"Group {group}: {', '.join(t.name for t in teams)}")
        self.schedule_group_stage()

# ---------------------------------------------------------------------- #
#                               MAIN PROGRAM                             #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    TEAMS = list(team_rankings.keys())
    STADIUMS = [
        "MetLife Stadium","AT&T Stadium","Arrowhead Stadium","Mercedes-Benz Stadium",
        "SoFi Stadium","Levi's Stadium","Hard Rock Stadium","NRG Stadium",
        "Gillette Stadium","Lumen Field","BC Place","BMO Field",
        "Estadio Azteca","Estadio BBVA","Estadio Akron"
    ]

    world_cup = WorldCup(TEAMS, STADIUMS)
    world_cup.start_tournament() 