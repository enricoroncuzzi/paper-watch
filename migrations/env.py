from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from paperwatch.config import get_settings
from paperwatch.store.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging, unless a caller (the eval) asked to keep its own
# output clean by setting the `configure_logger` attribute to False.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

# A test fixture or the eval sets `sqlalchemy.url` on the Config so migrations run against its own
# throwaway database; only when it is unset (empty in alembic.ini) does the app's Settings supply
# the URL. The value is re-escaped for ConfigParser so a percent-encoded password survives.
config.set_main_option(
    "sqlalchemy.url",
    (config.get_main_option("sqlalchemy.url") or get_settings().database_url).replace("%", "%%"),
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
