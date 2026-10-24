import type { components } from "@/lib/api-types";

export type RunSummary = components["schemas"]["RunSummary"];
export type RunDetailData = components["schemas"]["RunDetail"];
export type RunCounts = components["schemas"]["RunCounts"];
export type FollowupState = components["schemas"]["FollowupState"];
export type DeadLetterData = components["schemas"]["DeadLetterData"];
export type TodayData = components["schemas"]["TodayData"];
export type SuggestionsData = components["schemas"]["SuggestionsData"];
export type SuggestionCard = components["schemas"]["SuggestionCardData"];
export type SummaryData = components["schemas"]["SummaryData"];
export type SummaryFieldData = components["schemas"]["SummaryFieldData"];

export type Draft = components["schemas"]["DraftData"];
export type DraftDetail = components["schemas"]["DraftDetail"];
export type DraftsData = components["schemas"]["DraftsData"];
export type DraftFlag = components["schemas"]["DraftFlagData"];
/** The 503 body a failed draft generation returns. */
export type DraftErrorBody = components["schemas"]["DraftError"];

export type CorpusData = components["schemas"]["CorpusData"];
export type CorpusItem = components["schemas"]["CorpusItemData"];
export type CorpusItemDetail = components["schemas"]["CorpusItemDetail"];
export type TierChange = components["schemas"]["TierChangeData"];
export type SignalPoint = components["schemas"]["SignalPointData"];
export type ItemDraft = components["schemas"]["ItemDraftData"];

/**
 * The minimum a draft needs from an item: a pick's card, or a corpus item page. The dialog reads
 * only these, so an item that never reached the picks can still become a draft.
 */
export type DraftTarget = {
  item_id: string;
  title: string;
  tier: string;
  venue: string | null;
  org: string | null;
  profile: string | null;
  summary: SummaryData | null;
};
