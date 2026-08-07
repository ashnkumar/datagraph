"""Command line interface.

``datagraph demo`` runs the whole loop against synthetic data and shows its working: what was
retrieved, what redaction removed, what the answer was, how much each provider contributed,
and what they were paid. ``datagraph compare`` runs the same query under both attribution
engines side by side, which is the fastest way to see why the choice matters.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from datagraph.env import load_env_file
from datagraph.ledger import Ledger
from datagraph.marketplace import Marketplace, QueryResult
from datagraph.models import FakeModel, ModelClient
from datagraph.registry import Registry
from datagraph.sample_data import DEMO_QUESTION, seed_demo

__all__ = ["ConfigError", "main"]

ENGINES = ("shapley", "exact_shapley", "leave_one_out")
RESEARCHER = "rowan"
STARTING_CREDITS = 100_000
PAYMENT = 1_000


class ConfigError(Exception):
    """Something is wrong with the user's setup rather than with the run.

    Reported as a plain message with no exception class name in front of it, because the
    reader is being told what to go and fix.
    """


def _model(live: bool, console: Console) -> ModelClient:
    if not live:
        return FakeModel()

    # Checked here rather than left to the SDK, which raises an internal TypeError about
    # resolving an authentication method — accurate, but not an instruction.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ConfigError(
            "ANTHROPIC_API_KEY is not set, so --live cannot reach the API.\n"
            "Set it in a .env file (cp .env.example .env) or export it in your shell.\n"
            "Without --live everything runs offline against the deterministic model."
        )

    from datagraph.models import AnthropicModel

    console.print("[dim]Using the Anthropic API. Each coalition is one generation.[/dim]")
    return AnthropicModel()


def _market(engine: str, live: bool, console: Console) -> Marketplace:
    market = Marketplace(
        registry=seed_demo(Registry()),
        ledger=Ledger(),
        model=_model(live, console),
        engine=engine,
    )
    market.fund_researcher(RESEARCHER, STARTING_CREDITS)
    return market


def _sources_table(result: QueryResult) -> Table:
    table = Table(title="Retrieved records — raw vs. what the model was shown", expand=True)
    table.add_column("record", style="cyan", no_wrap=True)
    table.add_column("provider", style="magenta", no_wrap=True)
    table.add_column("suppressed by policy", style="red")
    table.add_column("disclosed to the model", style="green")

    for source in sorted(result.sources, key=lambda s: s.id):
        # Field *names* only — the view deliberately does not carry the removed values.
        table.add_row(
            source.id,
            source.provider_id,
            ", ".join(source.suppressed_fields) or "—",
            ", ".join(f"{k}={v}" for k, v in sorted(source.disclosed.items())),
        )
    return table


def _payout_table(result: QueryResult, market: Marketplace) -> Table:
    engine = result.attribution.engine if result.attribution else "n/a"
    table = Table(title=f"Contribution and payout — engine: {engine}", expand=True)
    table.add_column("provider", style="magenta", no_wrap=True)
    table.add_column("records used", style="cyan")
    table.add_column("measured share", justify="right")
    table.add_column("credits", justify="right", style="green")

    by_provider: dict[str, list[str]] = {}
    for source in result.sources:
        by_provider.setdefault(source.provider_id, []).append(source.id)

    for provider_id in sorted(by_provider):
        share = result.provider_weights.get(provider_id, 0.0)
        name = market.registry.get_provider(provider_id)
        table.add_row(
            f"{provider_id} ({name.name})" if name else provider_id,
            ", ".join(sorted(by_provider[provider_id])),
            f"{share:6.1%}",
            str(result.payouts.get(provider_id, 0)),
        )

    table.add_section()
    table.add_row("[bold]total[/bold]", "", "", f"[bold]{result.total_paid}[/bold]")
    return table


def _report(result: QueryResult, market: Marketplace, console: Console) -> None:
    console.print(Panel(result.question, title="Question", border_style="blue"))
    console.print(_sources_table(result))

    if result.refunded:
        console.print(
            Panel(
                f"{result.refund_reason}\n\nThe payment was returned in full. Nobody was paid.",
                title="Refunded",
                border_style="yellow",
            )
        )
        return

    console.print(Panel(result.answer, title="Answer", border_style="green"))
    console.print(_payout_table(result, market))

    attribution = result.attribution
    if attribution is not None:
        efficient = attribution.is_efficient
        console.print(
            f"[dim]{result.model_calls} model call(s). "
            f"Weights sum to {attribution.total_weight:.4f} of a possible "
            f"{attribution.grand_value:.4f} — "
            + (
                "[green]efficient: the payment is fully accounted for.[/green]"
                if efficient
                else "[red]not efficient: the shortfall was redistributed by normalization.[/red]"
            )
            + "[/dim]"
        )

    market.ledger.check_invariants()
    console.print(
        f"[dim]Ledger invariants hold. {market.ledger.credits_in_circulation()} credits in "
        f"circulation, {len(market.ledger.open_escrows())} escrow(s) open.[/dim]"
    )


def cmd_demo(args: argparse.Namespace, console: Console) -> int:
    market = _market(args.engine, args.live, console)
    result = market.query(RESEARCHER, args.question, args.payment)
    _report(result, market, console)
    return 0


def cmd_compare(args: argparse.Namespace, console: Console) -> int:
    """Run the same query under each engine and show where they disagree."""
    if args.live:
        console.print(
            f"[yellow]--live runs the query {len(ENGINES)} times, once per engine. "
            f"Each run costs up to 2^n generations.[/yellow]"
        )

    results: dict[str, QueryResult] = {}
    for engine in ENGINES:
        market = _market(engine, args.live, console)
        results[engine] = market.query(RESEARCHER, args.question, args.payment)

    console.print(Panel(args.question, title="Question", border_style="blue"))

    table = Table(title="Payout by attribution engine (credits)", expand=True)
    table.add_column("provider", style="magenta", no_wrap=True)
    for engine in ENGINES:
        table.add_column(engine, justify="right")

    providers = sorted({s.provider_id for r in results.values() for s in r.sources})
    for provider_id in providers:
        row = [provider_id]
        for engine in ENGINES:
            paid = results[engine].payouts.get(provider_id, 0)
            row.append(f"[red]{paid}[/red]" if paid == 0 else str(paid))
        table.add_row(*row)

    table.add_section()
    table.add_row(
        "[bold]weights sum to[/bold]",
        *(
            f"{results[e].attribution.total_weight:.4f}" if results[e].attribution else "—"
            for e in ENGINES
        ),
    )
    console.print(table)

    console.print(
        "\n[dim]Providers shown in [red]red[/red] earned nothing. Where two providers hold "
        "the same fact, removing either changes nothing, so leave-one-out scores both zero "
        "and their credits are reassigned to whoever happened to be unique. The Shapley "
        "columns split that credit instead, and their weights exhaust the payment.[/dim]"
    )
    return 0


def cmd_providers(args: argparse.Namespace, console: Console) -> int:
    market = _market(args.engine, live=False, console=console)

    table = Table(title="Registered providers", expand=True)
    table.add_column("id", style="magenta")
    table.add_column("name")
    table.add_column("records", justify="right")
    table.add_column("discloses")

    records = market.registry.all_records()
    for provider in market.registry.providers():
        owned = [r for r in records if r.provider_id == provider.id]
        fields = sorted({f for r in owned for f in r.disclosed})
        table.add_row(provider.id, provider.name, str(len(owned)), ", ".join(fields) or "—")

    console.print(table)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datagraph",
        description="A data marketplace that measures which data changed the answer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
        p.add_argument(
            "--payment", type=int, default=PAYMENT, help="credits to escrow for the query"
        )
        p.add_argument(
            "--live",
            action="store_true",
            help="call the Anthropic API instead of the deterministic offline model",
        )

    demo = sub.add_parser("demo", help="run one query end to end and show the working")
    add_common(demo)
    demo.add_argument("--engine", choices=ENGINES, default="shapley")
    demo.set_defaults(func=cmd_demo)

    compare = sub.add_parser("compare", help="run one query under every engine, side by side")
    add_common(compare)
    compare.set_defaults(func=cmd_compare, engine="shapley")

    providers = sub.add_parser("providers", help="list the seeded providers and what they share")
    providers.set_defaults(func=cmd_providers, engine="shapley", live=False)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    # Before anything reads the environment, and after argument parsing so that `--help`
    # never touches the filesystem. Anything already exported wins.
    load_env_file()

    try:
        return int(args.func(args, console))
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    except Exception as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
