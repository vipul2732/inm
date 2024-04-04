import click
import _run
import run_configurations as rc

@click.command()
@click.argument("name")
def main(name):
    _run.main(**rc.__dict__[name]._asdict())

if __name__ == "__main__":
    main()
