import dataclasses
import sys
import typing

import pandas as pd
from PyQt5.QtGui import QFont, QIntValidator, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

RANKINGS_FILE = "FantasyPositionRankings2025.csv"
ADP_FILE = "ppr_adp.csv"


@dataclasses.dataclass
class PlayerInfo:
    name: str
    adp: float
    rank: int
    notes: str


@dataclasses.dataclass
class AllPlayers:
    qbs: typing.List[PlayerInfo]
    rbs: typing.List[PlayerInfo]
    wrs: typing.List[PlayerInfo]
    tes: typing.List[PlayerInfo]


def get_players(rankings_file: str, adp_file: str) -> AllPlayers:
    df = pd.read_csv(rankings_file)
    adp_df = pd.read_csv(adp_file)
    adp_dict = adp_df.set_index("NAME")["ADP"].to_dict()
    qbs = []
    rbs = []
    wrs = []
    tes = []
    for index, row in df.iterrows():
        if pd.notna(row["QB"]):
            qbs.append(
                PlayerInfo(
                    name=row["QB"],
                    adp=adp_dict.get(row["QB"], float("inf")),
                    rank=len(qbs) + 1,
                    notes=row["QB_Notes"] if pd.notna(row["QB_Notes"]) else "",
                )
            )
        if pd.notna(row["RB"]):
            rbs.append(
                PlayerInfo(
                    name=row["RB"],
                    adp=adp_dict.get(row["RB"], float("inf")),
                    rank=len(rbs) + 1,
                    notes=row["RB_Notes"] if pd.notna(row["RB_Notes"]) else "",
                )
            )
        if pd.notna(row["WR"]):
            wrs.append(
                PlayerInfo(
                    name=row["WR"],
                    adp=adp_dict.get(row["WR"], float("inf")),
                    rank=len(wrs) + 1,
                    notes=row["WR_Notes"] if pd.notna(row["WR_Notes"]) else "",
                )
            )
        if pd.notna(row["TE"]):
            tes.append(
                PlayerInfo(
                    name=row["TE"],
                    adp=adp_dict.get(row["TE"], float("inf")),
                    rank=len(tes) + 1,
                    notes=row["TE_Notes"] if pd.notna(row["TE_Notes"]) else "",
                )
            )
    return AllPlayers(qbs=qbs, rbs=rbs, wrs=wrs, tes=tes)


# PyQt5 UI for displaying rankings
class RankingsApp(QWidget):
    def __init__(self, all_players: AllPlayers, num_teams: int, draft_position: int):
        super().__init__()
        self.setWindowTitle("Fantasy Rankings by Position")
        layout = QHBoxLayout()
        positions = [
            ("QB", all_players.qbs),
            ("RB", all_players.rbs),
            ("WR", all_players.wrs),
            ("TE", all_players.tes),
        ]
        self.num_teams = num_teams
        self.draft_position = draft_position
        self.all_players = all_players
        font = QFont()
        font.setPointSize(13)  # Increase font size

        self.num_teams = num_teams
        self.draft_position = draft_position
        self.current_pick = 1
        self.removed_stack = []  # Stack for undo functionality
        # Top button bar
        button_layout = QHBoxLayout()
        self.remove_btn = QPushButton("Remove Player")
        self.remove_btn.setStyleSheet("font-size:13pt; min-width:120px;")
        self.remove_btn.clicked.connect(self.remove_selected_player)
        button_layout.addWidget(self.remove_btn)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setStyleSheet("font-size:13pt; min-width:120px;")
        self.undo_btn.clicked.connect(self.undo_remove_player)
        button_layout.addWidget(self.undo_btn)
        self.skip_btn = QPushButton("Skip Pick")
        self.skip_btn.setStyleSheet("font-size:13pt; min-width:120px;")
        self.skip_btn.clicked.connect(self.skip_pick)
        button_layout.addWidget(self.skip_btn)
        self.skip_to_next_btn = QPushButton("Skip to next user pick")
        self.skip_to_next_btn.setStyleSheet("font-size:13pt; min-width:180px;")
        self.skip_to_next_btn.clicked.connect(self.skip_to_next_user_pick)
        button_layout.addWidget(self.skip_to_next_btn)
        # Main layout
        main_layout = QHBoxLayout()
        self.tables = []
        self.player_notes = {}  # (table, row) -> notes
        for pos_name, players_info in positions:
            vbox = QVBoxLayout()
            label = QLabel(pos_name)
            label.setFont(font)
            vbox.addWidget(label)
            table = QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Rank", "Player"])
            table.setRowCount(len(players_info))
            table.verticalHeader().hide()  # type: ignore
            for i, player_info in enumerate(players_info):
                rank_item = QTableWidgetItem(str(player_info.rank))
                rank_item.setFont(font)
                player_item = QTableWidgetItem(player_info.name)
                player_item.setFont(font)
                table.setItem(i, 0, rank_item)
                table.setItem(i, 1, player_item)
                self.player_notes[(table, i)] = player_info.notes
            table.resizeColumnsToContents()
            table.cellDoubleClicked.connect(self.show_player_notes)
            table.itemSelectionChanged.connect(self.handle_selection_changed)
            vbox.addWidget(table)
            main_layout.addLayout(vbox)
            self.tables.append(table)
        # Combine button bar and tables
        self.next_pick_label = QLabel()
        self.next_pick_label.setStyleSheet("font-size:13pt; margin-bottom:10px;")
        self.update_next_pick_label()
        container_layout = QVBoxLayout()
        container_layout.addLayout(button_layout)
        container_layout.addWidget(self.next_pick_label)
        container_layout.addLayout(main_layout)
        self.setLayout(container_layout)

    def show_player_notes(self, row, column):
        from PyQt5.QtGui import QTextOption
        from PyQt5.QtWidgets import QDialog, QPushButton, QTextEdit, QVBoxLayout

        # Find which table triggered the event
        sender = self.sender()
        notes = self.player_notes.get((sender, row), "")
        if notes:
            dlg = QDialog(self)
            dlg.setWindowTitle("Player Notes")
            dlg.resize(500, 400)
            layout = QVBoxLayout()
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setWordWrapMode(QTextOption.WordWrap)
            text_edit.setText(notes)
            text_edit.setStyleSheet("font-size:14pt;")
            layout.addWidget(text_edit)
            ok_btn = QPushButton("OK")
            ok_btn.setStyleSheet("font-size:14pt; min-width:80px;")
            ok_btn.clicked.connect(dlg.accept)
            layout.addWidget(ok_btn)
            dlg.setLayout(layout)
            dlg.exec_()

    def handle_selection_changed(self):
        sender = self.sender()
        for table in self.tables:
            if table is not sender:
                table.clearSelection()

    def remove_selected_player(self):
        for table in self.tables:
            selected = table.selectedItems()
            if selected:
                row = selected[0].row()
                rank = table.item(row, 0).text()
                name = table.item(row, 1).text()
                notes = self.player_notes.get((table, row), "")
                self.removed_stack.append(("remove", table, row, rank, name, notes))
                table.removeRow(row)
                self.current_pick += 1
                self.update_next_pick_label()
                break

    def undo_remove_player(self):
        if self.removed_stack:
            last_action = self.removed_stack.pop()
            if last_action[0] == "remove":
                _, table, row, rank, name, notes = last_action
                # Use the same font as the rest of the table
                font = QFont()
                font.setPointSize(13)
                table.insertRow(row)
                rank_item = QTableWidgetItem(rank)
                rank_item.setFont(font)
                player_item = QTableWidgetItem(name)
                player_item.setFont(font)
                table.setItem(row, 0, rank_item)
                table.setItem(row, 1, player_item)
                self.player_notes[(table, row)] = notes
                self.current_pick -= 1
            elif last_action[0] == "skip":
                self.current_pick -= 1
            self.update_next_pick_label()

    def skip_pick(self):
        self.removed_stack.append(("skip",))
        self.current_pick += 1
        self.update_next_pick_label()

    def skip_to_next_user_pick(self):
        from PyQt5.QtCore import QTimer

        self.cpu_picks_remaining = self.picks_until_next_user_pick()
        self.cpu_pick_queue = []
        # Precompute all CPU picks
        for _ in range(self.cpu_picks_remaining):
            available = []
            for table, (pos_name, pos_list) in zip(
                self.tables,
                [
                    ("QB", self.all_players.qbs),
                    ("RB", self.all_players.rbs),
                    ("WR", self.all_players.wrs),
                    ("TE", self.all_players.tes),
                ],
            ):
                for row in range(table.rowCount()):
                    item = table.item(row, 1)
                    name = item.text() if item else None
                    player = next(
                        (p for p in pos_list if getattr(p, "name", None) == name), None
                    )
                    adp = getattr(player, "adp", None) if player else None
                    if item and adp is not None:
                        available.append((name, adp, table, row))
            available.sort(key=lambda x: x[1])
            top_players = available[:10]
            if not top_players:
                break
            import random

            weights = [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.015, 0.005]
            weights = weights[: len(top_players)]
            weights = [w / sum(weights) for w in weights]
            pick_idx = random.choices(range(len(top_players)), weights)[0]
            name, adp, table, row = top_players[pick_idx]
            self.cpu_pick_queue.append((table, row))
        self.cpu_pick_timer = QTimer()
        self.cpu_pick_timer.setInterval(500)
        self.cpu_pick_timer.timeout.connect(self.make_cpu_pick)
        self.cpu_pick_timer.start()
        self.cpu_pick_highlighted = None
        self.make_cpu_highlight()  # Highlight first pick

    def make_cpu_highlight(self):
        # Remove previous highlight
        if self.cpu_pick_highlighted:
            table, row = self.cpu_pick_highlighted
            item = table.item(row, 1)
            if item:
                item.setBackground(QColor(255, 255, 255))
        # Highlight next pick
        if self.cpu_pick_queue:
            table, row = self.cpu_pick_queue[0]
            item = table.item(row, 1)
            if item:
                item.setBackground(QColor(102, 178, 255))  # Light blue highlight
            self.cpu_pick_highlighted = (table, row)
        else:
            self.cpu_pick_highlighted = None

    def make_cpu_pick(self):
        if not self.cpu_pick_queue or self.cpu_picks_remaining <= 0:
            self.cpu_pick_timer.stop()
            self.make_cpu_highlight()  # Remove highlight
            return
        table, row = self.cpu_pick_queue.pop(0)
        item = table.item(row, 1)
        rank = table.item(row, 0).text() if table.item(row, 0) else ""
        name = item.text() if item else ""
        notes = self.player_notes.get((table, row), "")
        if item:
            item.setBackground(QColor(255, 255, 255))  # Remove highlight before pick
        # Record CPU pick in undo stack
        self.removed_stack.append(("remove", table, row, rank, name, notes))
        table.removeRow(row)
        self.current_pick += 1
        self.cpu_picks_remaining -= 1
        self.update_next_pick_label()
        self.make_cpu_highlight()  # Highlight next pick

    def picks_until_next_user_pick(self):
        pick_in_round = self.current_pick % self.num_teams
        if pick_in_round == 0:
            pick_in_round = self.num_teams

        odd_round = ((self.current_pick - 1) // self.num_teams) % 2 == 0
        user_pick_this_round = (
            self.draft_position
            if odd_round
            else self.num_teams - self.draft_position + 1
        )
        user_pick_next_round = self.num_teams - user_pick_this_round + 1

        if user_pick_this_round >= pick_in_round:
            return user_pick_this_round - pick_in_round

        return self.num_teams - pick_in_round + user_pick_next_round

    def update_highlights(self):
        picks_next = self.picks_until_next_user_pick()
        if picks_next == 0:
            self.current_pick += 1
            picks_next = self.picks_until_next_user_pick()
            self.current_pick -= 1
        thresholds = [
            self.current_pick + picks_next,
            self.current_pick + 2 * picks_next,
            self.current_pick + 3 * picks_next,
        ]
        from PyQt5.QtGui import QColor

        color_map = [
            QColor(255, 102, 102),
            QColor(255, 255, 153),
            QColor(153, 255, 153),
        ]
        # For each table, only loop over existing rows
        for table, (pos_name, pos_list) in zip(
            self.tables,
            [
                ("QB", self.all_players.qbs),
                ("RB", self.all_players.rbs),
                ("WR", self.all_players.wrs),
                ("TE", self.all_players.tes),
            ],
        ):
            for row in range(table.rowCount()):
                item = table.item(row, 1)
                # Find the player name in the cell
                name = item.text() if item else None
                # Find the player info by name (assumes unique names per position)
                player = next(
                    (p for p in pos_list if getattr(p, "name", None) == name), None
                )
                adp = getattr(player, "adp", None) if player else None
                if item:
                    item.setBackground(QColor(255, 255, 255))  # Default: white
                    if adp is not None:
                        if adp <= thresholds[0]:
                            item.setBackground(color_map[0])
                        elif adp <= thresholds[1]:
                            item.setBackground(color_map[1])
                        elif adp <= thresholds[2]:
                            item.setBackground(color_map[2])

    def update_next_pick_label(self):
        picks = self.picks_until_next_user_pick()
        self.next_pick_label.setText(f"Picks until your next pick: {picks}")
        self.update_highlights()


class StartWindow(QWidget):
    def __init__(self, on_submit):
        super().__init__()
        self.setWindowTitle("Draft Setup")
        layout = QVBoxLayout()
        font = QFont()
        font.setPointSize(13)

        self.teams_label = QLabel("Teams in league (2-32):")
        self.teams_label.setFont(font)
        layout.addWidget(self.teams_label)
        self.teams_input = QLineEdit()
        self.teams_input.setValidator(QIntValidator(2, 32))
        self.teams_input.setFont(font)
        layout.addWidget(self.teams_input)

        self.pick_label = QLabel("Your pick (must be > 0 and <= teams in league):")
        self.pick_label.setFont(font)
        layout.addWidget(self.pick_label)
        self.pick_input = QLineEdit()
        self.pick_input.setValidator(QIntValidator(1, 32))
        self.pick_input.setFont(font)
        layout.addWidget(self.pick_input)

        self.enter_button = QPushButton("Enter")
        self.enter_button.setFont(font)
        self.enter_button.clicked.connect(self.validate_and_submit)
        layout.addWidget(self.enter_button)

        self.on_submit = on_submit
        self.setLayout(layout)

    def validate_and_submit(self):
        teams_text = self.teams_input.text()
        pick_text = self.pick_input.text()
        try:
            teams = int(teams_text)
            pick = int(pick_text)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers.")
            return
        if not (2 <= teams <= 32):
            QMessageBox.warning(
                self, "Input Error", "Teams in league must be between 2 and 32."
            )
            return
        if not (1 <= pick <= teams):
            QMessageBox.warning(
                self, "Input Error", "Your pick must be > 0 and < teams in league."
            )
            return
        self.on_submit(teams, pick)


if __name__ == "__main__":
    all_players = get_players(rankings_file=RANKINGS_FILE, adp_file=ADP_FILE)
    app = QApplication(sys.argv)

    def show_rankings(teams, pick):
        window.close()
        global rankings_window
        rankings_window = RankingsApp(all_players, teams, pick)
        rankings_window.show()

    window = StartWindow(show_rankings)
    window.show()
    sys.exit(app.exec_())
