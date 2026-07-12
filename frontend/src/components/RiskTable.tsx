"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getCustomerRisk,
  type CustomerRisk,
  type RiskLevel,
} from "@/lib/api";

const PAGE_SIZE = 25;

type FilterOption = "All" | RiskLevel;

function sortByProbabilityDesc(rows: CustomerRisk[]): CustomerRisk[] {
  return [...rows].sort((a, b) => b.churn_probability - a.churn_probability);
}

function RiskBadge({
  level,
  size = "sm",
}: {
  level: RiskLevel;
  size?: "sm" | "lg";
}) {
  const styles: Record<RiskLevel, string> = {
    High: "bg-red-50 text-red-700 ring-red-600/20",
    Medium: "bg-amber-50 text-amber-800 ring-amber-600/20",
    Low: "bg-slate-100 text-slate-600 ring-slate-500/20",
  };
  const sizeClass =
    size === "lg"
      ? "px-3 py-1 text-sm font-semibold"
      : "px-2 py-0.5 text-xs font-medium";

  return (
    <span
      className={`inline-flex items-center rounded-md ring-1 ring-inset ${styles[level]} ${sizeClass}`}
    >
      {level}
    </span>
  );
}

function suggestedAction(customer: CustomerRisk): string {
  const reasonsText = customer.top_3_reasons.join(" ").toLowerCase();
  const mentionsComplaint = reasonsText.includes("complaint");
  const mentionsTenureOrOrder =
    reasonsText.includes("tenure") ||
    reasonsText.includes("order") ||
    reasonsText.includes("last order");

  if (customer.risk_level === "High" && mentionsComplaint) {
    return "Escalate to account manager and offer service resolution";
  }
  if (customer.risk_level === "High" && mentionsTenureOrOrder) {
    return "Offer retention discount or loyalty incentive";
  }
  if (customer.risk_level === "High") {
    return "Prioritize outreach within 24 hours with a personalized retention offer";
  }
  if (customer.risk_level === "Medium") {
    return "Add to weekly check-in list";
  }
  return "Monitor engagement; no immediate intervention required";
}

function CustomerDetailModal({
  customer,
  onClose,
}: {
  customer: CustomerRisk;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
      onClick={onClose}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="customer-detail-title"
        className="relative w-full max-w-lg rounded-xl border border-slate-200 bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          aria-label="Close"
          className="absolute right-3 top-3 rounded-md p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          onClick={onClose}
        >
          <span className="block text-lg leading-none">×</span>
        </button>

        <h3
          id="customer-detail-title"
          className="pr-8 text-xl font-semibold tabular-nums text-slate-900"
        >
          Customer {customer.customer_id}
        </h3>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <RiskBadge level={customer.risk_level} size="lg" />
          <p className="text-2xl font-semibold tabular-nums text-slate-900">
            {(customer.churn_probability * 100).toFixed(1)}%
            <span className="ml-1 text-sm font-medium text-slate-500">
              churn probability
            </span>
          </p>
        </div>

        <div className="mt-6">
          <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            Top reasons
          </h4>
          <ul className="mt-2 space-y-2">
            {customer.top_3_reasons.map((reason) => (
              <li
                key={reason}
                className="flex items-start gap-2 text-sm text-slate-800"
              >
                <span
                  aria-hidden="true"
                  className="mt-1.5 size-1.5 shrink-0 rounded-full bg-slate-400"
                />
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-6 rounded-lg border border-slate-200 bg-slate-50 p-4">
          <h4 className="text-sm font-semibold text-slate-700">
            Suggested Action
          </h4>
          <p className="mt-1 text-sm leading-relaxed text-slate-700">
            {suggestedAction(customer)}
          </p>
        </div>
      </div>
    </div>
  );
}

export default function RiskTable() {
  const [rows, setRows] = useState<CustomerRisk[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [includeLow, setIncludeLow] = useState(false);
  const [filter, setFilter] = useState<FilterOption>("All");
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<CustomerRisk | null>(null);

  const loadRows = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let next: CustomerRisk[];

      if (filter === "All") {
        const levels: RiskLevel[] = includeLow
          ? ["High", "Medium", "Low"]
          : ["High", "Medium"];
        const batches = await Promise.all(
          levels.map((level) => getCustomerRisk(level)),
        );
        next = sortByProbabilityDesc(batches.flat());
      } else {
        next = sortByProbabilityDesc(await getCustomerRisk(filter));
      }

      setRows(next);
      setPage(1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load customers");
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, [filter, includeLow]);

  useEffect(() => {
    void loadRows();
  }, [loadRows]);

  const searchActive = search.trim().length > 0;

  const filteredRows = useMemo(() => {
    const query = search.trim();
    if (!query) return rows;
    return rows.filter((row) => String(row.customer_id).includes(query));
  }, [rows, search]);

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);

  const displayRows = useMemo(() => {
    if (searchActive) return filteredRows;
    const start = (currentPage - 1) * PAGE_SIZE;
    return filteredRows.slice(start, start + PAGE_SIZE);
  }, [filteredRows, currentPage, searchActive]);

  const rangeStart =
    filteredRows.length === 0
      ? 0
      : searchActive
        ? 1
        : (currentPage - 1) * PAGE_SIZE + 1;
  const rangeEnd = searchActive
    ? filteredRows.length
    : Math.min(currentPage * PAGE_SIZE, filteredRows.length);

  return (
    <div className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-col gap-3 border-b border-slate-200 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
        <label className="flex items-center gap-2 text-sm text-slate-700">
          <input
            type="checkbox"
            className="size-4 rounded border-slate-300 text-slate-800 focus:ring-slate-400"
            checked={includeLow}
            disabled={filter !== "All"}
            onChange={(e) => setIncludeLow(e.target.checked)}
          />
          Include Low risk customers
        </label>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div className="relative">
            <input
              type="text"
              inputMode="numeric"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              placeholder="Search by Customer ID..."
              className="w-full rounded-md border border-slate-300 bg-white py-1.5 pl-3 pr-8 text-sm text-slate-800 shadow-sm placeholder:text-slate-400 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400 sm:w-56"
              aria-label="Search by Customer ID"
            />
            {search && (
              <button
                type="button"
                aria-label="Clear search"
                className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                onClick={() => setSearch("")}
              >
                ×
              </button>
            )}
          </div>

          <label className="flex items-center gap-2 text-sm text-slate-700">
            <span className="whitespace-nowrap">Risk level</span>
            <select
              className="rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-800 shadow-sm focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
              value={filter}
              onChange={(e) => setFilter(e.target.value as FilterOption)}
            >
              <option value="All">All</option>
              <option value="High">High</option>
              <option value="Medium">Medium</option>
              <option value="Low">Low</option>
            </select>
          </label>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-4 py-3 font-medium text-slate-600">
                Customer ID
              </th>
              <th className="px-4 py-3 font-medium text-slate-600">
                Risk Level
              </th>
              <th className="px-4 py-3 font-medium text-slate-600">
                Churn Probability
              </th>
              <th className="px-4 py-3 font-medium text-slate-600">
                Top Reasons
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {loading && (
              <tr>
                <td
                  colSpan={4}
                  className="px-4 py-8 text-center text-slate-500"
                >
                  Loading customers…
                </td>
              </tr>
            )}
            {!loading && error && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-red-600">
                  {error}
                </td>
              </tr>
            )}
            {!loading && !error && displayRows.length === 0 && (
              <tr>
                <td
                  colSpan={4}
                  className="px-4 py-8 text-center text-slate-500"
                >
                  {searchActive
                    ? "No customers match this search."
                    : "No customers match this filter."}
                </td>
              </tr>
            )}
            {!loading &&
              !error &&
              displayRows.map((row, index) => (
                <tr
                  key={row.customer_id}
                  className={`cursor-pointer transition-colors ${
                    index % 2 === 0 ? "bg-white" : "bg-slate-50/70"
                  } hover:bg-slate-100`}
                  onClick={() => setSelected(row)}
                >
                  <td className="whitespace-nowrap px-4 py-3 font-medium tabular-nums text-slate-900">
                    {row.customer_id}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3">
                    <RiskBadge level={row.risk_level} />
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 tabular-nums text-slate-800">
                    {(row.churn_probability * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1.5">
                      {row.top_3_reasons.map((reason) => (
                        <span
                          key={reason}
                          className="inline-flex max-w-full rounded-md bg-slate-100 px-2 py-0.5 text-xs text-slate-700"
                        >
                          {reason}
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      <div className="flex flex-col gap-3 border-t border-slate-200 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-slate-600">
          {searchActive
            ? `Showing ${filteredRows.length.toLocaleString()} match${filteredRows.length === 1 ? "" : "es"}`
            : `Showing ${rangeStart}-${rangeEnd} of ${filteredRows.length.toLocaleString()}`}
        </p>
        {!searchActive && (
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
              disabled={loading || currentPage <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              Previous
            </button>
            <span className="text-sm tabular-nums text-slate-500">
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
              disabled={loading || currentPage >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              Next
            </button>
          </div>
        )}
      </div>

      {selected && (
        <CustomerDetailModal
          customer={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
