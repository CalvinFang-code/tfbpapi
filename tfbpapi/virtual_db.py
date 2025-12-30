"""
VirtualDB provides a unified query interface across heterogeneous datasets.

This module enables cross-dataset queries with standardized field names and values,
mapping varying experimental condition structures to a common schema through external
YAML configuration.

Key Components:
- VirtualDB: Main interface for unified cross-dataset queries
- Helper functions: get_nested_value(), normalize_value() for metadata extraction
- Configuration-driven schema via models.MetadataConfig

Example Usage:
    >>> from tfbpapi.datainfo import VirtualDB
    >>> vdb = VirtualDB("config.yaml")
    >>>
    >>> # Discover available fields
    >>> fields = vdb.get_fields()
    >>> print(fields)  # ["carbon_source", "temperature_celsius", ...]
    >>>
    >>> # Query across datasets
    >>> df = vdb.query(
    ...     filters={"carbon_source": "glucose", "temperature_celsius": 30},
    ...     fields=["sample_id", "carbon_source", "temperature_celsius"]
    ... )
    >>>
    >>> # Get complete data with measurements
    >>> df = vdb.query(
    ...     filters={"carbon_source": "glucose"},
    ...     complete=True
    ... )

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from tfbpapi.datacard import DataCard
from tfbpapi.errors import DataCardError
from tfbpapi.hf_cache_manager import HfCacheManager
from tfbpapi.models import MetadataConfig, PropertyMapping


def get_nested_value(data: dict, path: str) -> Any:
    """
    Navigate nested dict/list using dot notation.

    Handles missing intermediate keys gracefully by returning None.
    Supports extracting properties from lists of dicts.

    :param data: Dictionary to navigate
    :param path: Dot-separated path (e.g., "media.carbon_source.compound")
    :return: Value at path or None if not found

    Examples:
        Simple nested dict:
            get_nested_value({"media": {"name": "YPD"}}, "media.name")
            Returns: "YPD"

        List of dicts - extract property from each item:
            get_nested_value(
                {
                    "media": {
                        "carbon_source": [
                            {"compound": "glucose"},
                            {"compound": "galactose"},
                        ]
                    }
                },
                "media.carbon_source.compound",
            )
            Returns: ["glucose", "galactose"]

    """
    if not isinstance(data, dict):
        return None

    keys = path.split(".")
    current = data

    for i, key in enumerate(keys):
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
        elif isinstance(current, list):
            # If current is a list and we have more keys,
            # extract property from each item
            if i < len(keys):
                # Extract the remaining path from each list item
                remaining_path = ".".join(keys[i:])
                results = []
                for item in current:
                    if isinstance(item, dict):
                        val = get_nested_value(item, remaining_path)
                        if val is not None:
                            results.append(val)
                return results if results else None
        else:
            return None

    return current


def normalize_value(
    actual_value: Any,
    aliases: dict[str, list[Any]] | None,
    missing_value_label: str | None = None,
) -> str:
    """
    Normalize a value using optional alias mappings (case-insensitive).

    Returns the alias name if a match is found, otherwise returns the
    original value as a string. Handles missing values by returning
    the configured missing_value_label.

    :param actual_value: The value from the data to normalize
    :param aliases: Optional dict mapping alias names to lists of actual values.
                    Example: {"glucose": ["D-glucose", "dextrose"]}
    :param missing_value_label: Label to use for None/missing values
    :return: Alias name if match found, missing_value_label if None,
             otherwise str(actual_value)

    Examples:
        With aliases - exact match:
            normalize_value("D-glucose", {"glucose": ["D-glucose", "dextrose"]})
            Returns: "glucose"

        With aliases - case-insensitive match:
            normalize_value("DEXTROSE", {"glucose": ["D-glucose", "dextrose"]})
            Returns: "glucose"

        Missing value:
            normalize_value(None, None, "unspecified")
            Returns: "unspecified"

        No alias match - pass through:
            normalize_value("maltose", {"glucose": ["D-glucose"]})
            Returns: "maltose"

    """
    # Handle None/missing values
    if actual_value is None:
        return missing_value_label if missing_value_label else "None"

    if aliases is None:
        return str(actual_value)

    # Convert to string for comparison (case-insensitive)
    actual_str = str(actual_value).lower()

    # Check each alias mapping
    for alias_name, actual_values in aliases.items():
        for val in actual_values:
            if str(val).lower() == actual_str:
                return alias_name

    # No match found - pass through original value
    return str(actual_value)


class VirtualDB:
    """
    Unified query interface across heterogeneous datasets.

    VirtualDB provides a virtual database layer over multiple HuggingFace datasets,
    allowing cross-dataset queries with standardized field names and normalized values.
    Each configured dataset becomes a view with a common schema defined by external
    YAML configuration.

    The YAML configuration specifies:
    1. Property mappings: How to extract each field from dataset structures
    2. Factor aliases: Normalize varying terminologies to standard values
    3. Missing value labels: Handle missing data consistently
    4. Descriptions: Document each field's semantics

    Attributes:
        config: MetadataConfig instance with all configuration
        token: Optional HuggingFace token for private datasets
        cache: Dict mapping (repo_id, config_name) to cached DataFrame views

    """

    def __init__(self, config_path: Path | str, token: str | None = None):
        """
        Initialize VirtualDB with configuration and optional auth token.

        :param config_path: Path to YAML configuration file
        :param token: Optional HuggingFace token for private datasets
        :raises FileNotFoundError: If config file doesn't exist
        :raises ValueError: If configuration is invalid

        """
        self.config = MetadataConfig.from_yaml(config_path)
        self.token = token
        self.cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get_fields(
        self, repo_id: str | None = None, config_name: str | None = None
    ) -> list[str]:
        """
        Get list of queryable fields.

        :param repo_id: Optional repository ID to filter to specific dataset
        :param config_name: Optional config name (required if repo_id provided)
        :return: List of field names

        Examples:
            All fields across all datasets:
                fields = vdb.get_fields()

            Fields for specific dataset:
                fields = vdb.get_fields("BrentLab/harbison_2004", "harbison_2004")

        """
        if repo_id is not None and config_name is not None:
            # Get fields for specific dataset
            mappings = self.config.get_property_mappings(repo_id, config_name)
            return sorted(mappings.keys())

        if repo_id is not None or config_name is not None:
            raise ValueError(
                "Both repo_id and config_name must be provided, or neither"
            )

        # Get all fields across all datasets
        all_fields: set[str] = set()
        for repo_id, repo_config in self.config.repositories.items():
            # Add repo-wide fields
            all_fields.update(repo_config.properties.keys())
            # Add dataset-specific fields
            if repo_config.dataset:
                for dataset_props in repo_config.dataset.values():
                    all_fields.update(dataset_props.keys())

        return sorted(all_fields)

    def get_common_fields(self) -> list[str]:
        """
        Get fields present in ALL configured datasets.

        :return: List of field names common to all datasets

        Example:
            common = vdb.get_common_fields()
            # ["carbon_source", "temperature_celsius"]

        """
        if not self.config.repositories:
            return []

        # Get field sets for each dataset
        dataset_fields: list[set[str]] = []
        for repo_id, repo_config in self.config.repositories.items():
            if repo_config.dataset:
                for config_name in repo_config.dataset.keys():
                    mappings = self.config.get_property_mappings(repo_id, config_name)
                    dataset_fields.append(set(mappings.keys()))

        if not dataset_fields:
            return []

        # Return intersection
        common = set.intersection(*dataset_fields)
        return sorted(common)

    def get_unique_values(
        self, field: str, by_dataset: bool = False
    ) -> list[str] | dict[str, list[str]]:
        """
        Get unique values for a field across datasets (with normalization).

        :param field: Field name to get values for
        :param by_dataset: If True, return dict keyed by dataset identifier
        :return: List of unique normalized values, or dict if by_dataset=True

        Examples:
            All unique values:
                values = vdb.get_unique_values("carbon_source")
                # ["glucose", "galactose", "raffinose"]

            Values by dataset:
                values = vdb.get_unique_values("carbon_source", by_dataset=True)
                # {"BrentLab/harbison_2004": ["glucose", "galactose"],
                #  "BrentLab/kemmeren_2014": ["glucose", "raffinose"]}

        """
        if by_dataset:
            result: dict[str, list[str]] = {}
        else:
            all_values: set[str] = set()

        # Query each dataset that has this field
        for repo_id, repo_config in self.config.repositories.items():
            if repo_config.dataset:
                for config_name in repo_config.dataset.keys():
                    mappings = self.config.get_property_mappings(repo_id, config_name)
                    if field not in mappings:
                        continue

                    # Build metadata table for this dataset
                    metadata_df = self._build_metadata_table(repo_id, config_name)
                    if metadata_df.empty or field not in metadata_df.columns:
                        continue

                    # Get unique values (already normalized)
                    unique_vals = metadata_df[field].dropna().unique().tolist()

                    if by_dataset:
                        dataset_key = f"{repo_id}/{config_name}"
                        result[dataset_key] = sorted(unique_vals)
                    else:
                        all_values.update(unique_vals)

        if by_dataset:
            return result
        else:
            return sorted(all_values)

    def query(
        self,
        filters: dict[str, Any] | None = None,
        datasets: list[tuple[str, str]] | None = None,
        fields: list[str] | None = None,
        complete: bool = False,
    ) -> pd.DataFrame:
        """
        Query VirtualDB with optional filters and field selection.

        :param filters: Dict of field:value pairs to filter on
        :param datasets: List of (repo_id, config_name) tuples to query (None = all)
        :param fields: List of field names to return (None = all)
        :param complete: If True, return measurement-level data; if False, sample-level
        :return: DataFrame with query results

        Examples:
            Basic query across all datasets:
                df = vdb.query(filters={"carbon_source": "glucose"})

            Query specific datasets with field selection:
                df = vdb.query(
                    filters={"carbon_source": "glucose", "temperature_celsius": 30},
                    datasets=[("BrentLab/harbison_2004", "harbison_2004")],
                    fields=["sample_id", "carbon_source", "temperature_celsius"]
                )

            Complete data with measurements:
                df = vdb.query(
                    filters={"carbon_source": "glucose"},
                    complete=True
                )

        """
        # Determine which datasets to query
        if datasets is None:
            # Query all configured datasets
            datasets = []
            for repo_id, repo_config in self.config.repositories.items():
                if repo_config.dataset:
                    for config_name in repo_config.dataset.keys():
                        datasets.append((repo_id, config_name))

        if not datasets:
            return pd.DataFrame()

        # Query each dataset
        results: list[pd.DataFrame] = []
        for repo_id, config_name in datasets:
            # Build metadata table
            metadata_df = self._build_metadata_table(repo_id, config_name)
            if metadata_df.empty:
                continue

            # Apply filters
            if filters:
                metadata_df = self._apply_filters(
                    metadata_df, filters, repo_id, config_name
                )

            # If complete=True, join with full data
            if complete:
                sample_ids = metadata_df["sample_id"].tolist()
                if sample_ids:
                    full_df = self._get_complete_data(
                        repo_id, config_name, sample_ids, metadata_df
                    )
                    if not full_df.empty:
                        metadata_df = full_df

            # Select requested fields
            if fields:
                # Keep sample_id and any dataset identifier columns
                keep_cols = ["sample_id"]
                if "dataset_id" in metadata_df.columns:
                    keep_cols.append("dataset_id")
                # Add requested fields that exist
                for field in fields:
                    if field in metadata_df.columns and field not in keep_cols:
                        keep_cols.append(field)
                metadata_df = metadata_df[keep_cols].copy()

            # Add dataset identifier (ensure copy before modifying)
            if "dataset_id" not in metadata_df.columns:
                metadata_df = metadata_df.copy()
                metadata_df["dataset_id"] = f"{repo_id}/{config_name}"

            results.append(metadata_df)

        if not results:
            return pd.DataFrame()

        # Concatenate results, filling NaN for missing columns
        return pd.concat(results, ignore_index=True, sort=False)

    def materialize_views(self, datasets: list[tuple[str, str]] | None = None) -> None:
        """
        Build and cache metadata DataFrames for faster subsequent queries.

        :param datasets: List of (repo_id, config_name) tuples to materialize
                        (None = materialize all)

        Example:
            vdb.materialize_views()  # Cache all datasets
            vdb.materialize_views([("BrentLab/harbison_2004", "harbison_2004")])

        """
        if datasets is None:
            # Materialize all configured datasets
            datasets = []
            for repo_id, repo_config in self.config.repositories.items():
                if repo_config.dataset:
                    for config_name in repo_config.dataset.keys():
                        datasets.append((repo_id, config_name))

        for repo_id, config_name in datasets:
            # Build and cache
            self._build_metadata_table(repo_id, config_name, use_cache=False)

    def invalidate_cache(self, datasets: list[tuple[str, str]] | None = None) -> None:
        """
        Clear cached metadata DataFrames.

        :param datasets: List of (repo_id, config_name) tuples to invalidate
                        (None = invalidate all)

        Example:
            vdb.invalidate_cache()  # Clear all cache
            vdb.invalidate_cache([("BrentLab/harbison_2004", "harbison_2004")])

        """
        if datasets is None:
            self.cache.clear()
        else:
            for dataset_key in datasets:
                if dataset_key in self.cache:
                    del self.cache[dataset_key]

    def _build_metadata_table(
        self, repo_id: str, config_name: str, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Build metadata table for a single dataset.

        Extracts sample-level metadata from experimental conditions hierarchy and field
        definitions, with normalization and missing value handling.

        :param repo_id: Repository ID
        :param config_name: Configuration name
        :param use_cache: Whether to use/update cache
        :return: DataFrame with one row per sample_id

        """
        cache_key = (repo_id, config_name)

        # Check cache
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Load DataCard and CacheManager
            card = DataCard(repo_id, token=self.token)
            cache_mgr = HfCacheManager(
                repo_id, duckdb_conn=duckdb.connect(":memory:"), token=self.token
            )

            # Get property mappings
            property_mappings = self.config.get_property_mappings(repo_id, config_name)
            if not property_mappings:
                return pd.DataFrame()

            # Extract repo/config-level metadata
            repo_metadata = self._extract_repo_level(
                card, config_name, property_mappings
            )

            # Extract field-level metadata
            field_metadata = self._extract_field_level(
                card, config_name, property_mappings
            )

            # Get sample-level data from HuggingFace
            config = card.get_config(config_name)
            if config and hasattr(config, "metadata_fields") and config.metadata_fields:
                # Select only metadata fields
                columns = ", ".join(config.metadata_fields)
                if "sample_id" not in config.metadata_fields:
                    columns = f"sample_id, {columns}"
                sql = f"SELECT DISTINCT {columns} FROM {config_name}"
            else:
                # No metadata_fields specified, select all
                sql = f"SELECT DISTINCT * FROM {config_name}"

            df = cache_mgr.query(sql, config_name)

            # One row per sample_id
            if "sample_id" in df.columns:
                df = df.groupby("sample_id").first().reset_index()

            # Add repo-level metadata as columns
            for prop_name, values in repo_metadata.items():
                # Use first value (repo-level properties are constant)
                df[prop_name] = values[0] if values else None

            # Add field-level metadata
            if field_metadata:
                df = self._add_field_metadata(df, field_metadata)

            # Cache result
            if use_cache:
                self.cache[cache_key] = df

            return df

        except Exception:
            # Return empty DataFrame on error
            return pd.DataFrame()

    def _extract_repo_level(
        self,
        card: DataCard,
        config_name: str,
        property_mappings: dict[str, PropertyMapping],
    ) -> dict[str, list[str]]:
        """
        Extract and normalize repo/config-level metadata.

        :param card: DataCard instance
        :param config_name: Configuration name
        :param property_mappings: Property mappings for this dataset
        :return: Dict mapping property names to normalized values

        """
        metadata: dict[str, list[str]] = {}

        # Get experimental conditions
        try:
            conditions = card.get_experimental_conditions(config_name)
        except DataCardError:
            conditions = {}

        if not conditions:
            return metadata

        # Extract each mapped property
        for prop_name, mapping in property_mappings.items():
            # Skip field-level mappings
            if mapping.field is not None:
                continue

            # Build full path
            full_path = mapping.path

            # Skip if path is None (shouldn't happen for repo-level, but be safe)
            if full_path is None:
                continue

            # Get value at path
            value = get_nested_value(conditions, full_path)

            # Handle missing values
            missing_label = self.config.missing_value_labels.get(prop_name)
            if value is None:
                if missing_label:
                    metadata[prop_name] = [missing_label]
                continue

            # Ensure value is a list
            actual_values = [value] if not isinstance(value, list) else value

            # Normalize using aliases
            aliases = self.config.factor_aliases.get(prop_name)
            normalized_values = [
                normalize_value(v, aliases, missing_label) for v in actual_values
            ]

            metadata[prop_name] = normalized_values

        return metadata

    def _extract_field_level(
        self,
        card: DataCard,
        config_name: str,
        property_mappings: dict[str, PropertyMapping],
    ) -> dict[str, dict[str, Any]]:
        """
        Extract and normalize field-level metadata.

        :param card: DataCard instance
        :param config_name: Configuration name
        :param property_mappings: Property mappings for this dataset
        :return: Dict mapping field values to their normalized metadata

        """
        field_metadata: dict[str, dict[str, Any]] = {}

        # Group property mappings by field
        field_mappings: dict[str, dict[str, str | None]] = {}
        for prop_name, mapping in property_mappings.items():
            if mapping.field is not None:
                field_name = mapping.field
                if field_name not in field_mappings:
                    field_mappings[field_name] = {}
                # Store path (can be None for column aliases)
                field_mappings[field_name][prop_name] = mapping.path

        # Process each field that has mappings
        for field_name, prop_paths in field_mappings.items():
            # Get field definitions
            definitions = card.get_field_definitions(config_name, field_name)
            if not definitions:
                continue

            # Extract metadata for each field value
            for field_value, definition in definitions.items():
                if field_value not in field_metadata:
                    field_metadata[field_value] = {}

                for prop_name, path in prop_paths.items():
                    # Handle path=None case: use field_value directly
                    if path is None:
                        value = field_value
                    else:
                        # Get value at path
                        value = get_nested_value(definition, path)

                    # Handle missing values
                    missing_label = self.config.missing_value_labels.get(prop_name)
                    if value is None:
                        if missing_label:
                            field_metadata[field_value][prop_name] = [missing_label]
                        continue

                    # Ensure value is a list
                    actual_values = [value] if not isinstance(value, list) else value

                    # Normalize using aliases
                    aliases = self.config.factor_aliases.get(prop_name)
                    normalized_values = [
                        normalize_value(v, aliases, missing_label)
                        for v in actual_values
                    ]

                    field_metadata[field_value][prop_name] = normalized_values

        return field_metadata

    def _add_field_metadata(
        self, df: pd.DataFrame, field_metadata: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Add columns from field-level metadata to DataFrame.

        :param df: DataFrame with base sample metadata
        :param field_metadata: Dict mapping field values to their properties
        :return: DataFrame with additional property columns

        """
        # For each field value, add its properties as columns
        for field_value, properties in field_metadata.items():
            for prop_name, prop_values in properties.items():
                # Initialize column if needed
                if prop_name not in df.columns:
                    df[prop_name] = None

                # Find rows where any column matches field_value
                for col in df.columns:
                    if col in [prop_name, "sample_id", "dataset_id"]:
                        continue
                    mask = df[col] == field_value
                    if mask.any():
                        # Set property value (take first from list)
                        value = prop_values[0] if prop_values else None
                        df.loc[mask, prop_name] = value

        return df

    def _apply_filters(
        self,
        df: pd.DataFrame,
        filters: dict[str, Any],
        repo_id: str,
        config_name: str,
    ) -> pd.DataFrame:
        """
        Apply filters to DataFrame with alias expansion and numeric handling.

        :param df: DataFrame to filter
        :param filters: Dict of field:value pairs
        :param repo_id: Repository ID (for alias lookup)
        :param config_name: Config name (for alias lookup)
        :return: Filtered DataFrame

        """
        for field, filter_value in filters.items():
            if field not in df.columns:
                continue

            # Handle numeric range filters
            if isinstance(filter_value, tuple):
                operator = filter_value[0]
                # For numeric comparisons, try to convert column to numeric
                # (normalize_value returns strings,
                # but we need numeric for range queries)
                try:
                    df_field = pd.to_numeric(df[field], errors="coerce")
                except (ValueError, TypeError):
                    df_field = df[field]

                if operator == "between" and len(filter_value) == 3:
                    df = df[
                        (df_field >= filter_value[1]) & (df_field <= filter_value[2])
                    ]
                elif operator in (">=", ">", "<=", "<", "==", "!="):
                    if operator == ">=":
                        df = df[df_field >= filter_value[1]]
                    elif operator == ">":
                        df = df[df_field > filter_value[1]]
                    elif operator == "<=":
                        df = df[df_field <= filter_value[1]]
                    elif operator == "<":
                        df = df[df_field < filter_value[1]]
                    elif operator == "==":
                        df = df[df_field == filter_value[1]]
                    elif operator == "!=":
                        df = df[df_field != filter_value[1]]
            else:
                # Exact match with alias expansion
                aliases = self.config.factor_aliases.get(field)
                if aliases:
                    # Expand filter value to all aliases
                    expanded_values = [filter_value]
                    for alias_name, actual_values in aliases.items():
                        if alias_name == filter_value:
                            # Add all actual values for this alias
                            expanded_values.extend([str(v) for v in actual_values])
                    df = df[df[field].isin(expanded_values)]
                else:
                    # No aliases, exact match
                    # Handle type conversion: normalize_value returns strings,
                    # so convert filter_value to string for comparison
                    df = df[df[field] == str(filter_value)]

        return df.copy()

    def _get_complete_data(
        self,
        repo_id: str,
        config_name: str,
        sample_ids: list[str],
        metadata_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get complete data (with measurements) for sample_ids.

        Uses WHERE sample_id IN (...) approach for efficient retrieval.

        :param repo_id: Repository ID
        :param config_name: Configuration name
        :param sample_ids: List of sample IDs to retrieve
        :param metadata_df: Metadata DataFrame to merge with
        :return: DataFrame with measurements and metadata

        """
        try:
            cache_mgr = HfCacheManager(
                repo_id, duckdb_conn=duckdb.connect(":memory:"), token=self.token
            )

            # Build IN clause
            sample_id_list = ", ".join([f"'{sid}'" for sid in sample_ids])
            sql = f"""
                SELECT *
                FROM {config_name}
                WHERE sample_id IN ({sample_id_list})
            """

            full_df = cache_mgr.query(sql, config_name)

            # Merge with metadata (metadata_df has normalized fields)
            # Drop metadata columns from full_df to avoid duplicates
            metadata_cols = [
                col
                for col in metadata_df.columns
                if col not in ["sample_id", "dataset_id"]
            ]
            full_df = full_df.drop(
                columns=[c for c in metadata_cols if c in full_df.columns],
                errors="ignore",
            )

            # Merge on sample_id
            result = full_df.merge(metadata_df, on="sample_id", how="left")

            return result

        except Exception:
            return pd.DataFrame()

    def __repr__(self) -> str:
        """String representation."""
        n_repos = len(self.config.repositories)
        n_datasets = sum(
            len(rc.dataset) if rc.dataset else 0
            for rc in self.config.repositories.values()
        )
        n_cached = len(self.cache)
        return (
            f"VirtualDB({n_repos} repositories, {n_datasets} datasets configured, "
            f"{n_cached} views cached)"
        )
