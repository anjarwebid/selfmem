"""Stripe billing.

All public functions are no-ops (or return 503-equivalent) when STRIPE_SECRET_KEY
is unset, so the same code path is safe in selfhosted mode where Stripe is
disabled. The `IS_SAAS` env flag controls UI exposure; this module just handles
the wire protocol.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import config
import db

log = logging.getLogger(__name__)


# Lazy-import stripe so the package is optional at import time.
def _stripe():
    import stripe as _s
    if not config.STRIPE_SECRET_KEY:
        raise RuntimeError("Stripe is not configured (STRIPE_SECRET_KEY missing)")
    _s.api_key = config.STRIPE_SECRET_KEY
    return _s


def is_configured() -> bool:
    return bool(config.STRIPE_SECRET_KEY and config.STRIPE_UNLIMITED_PRICE_ID)


async def create_checkout_session(org: dict, success_url: str, cancel_url: str) -> str:
    """Return the Stripe Checkout URL for upgrading this org to Unlimited."""
    s = _stripe()
    params = {
        "mode": "subscription",
        "line_items": [{"price": config.STRIPE_UNLIMITED_PRICE_ID, "quantity": 1}],
        "success_url": success_url,
        "cancel_url": cancel_url,
        "client_reference_id": org["id"],
        "metadata": {"org_id": org["id"], "org_slug": org["slug"]},
        "allow_promotion_codes": True,
    }
    if org.get("stripe_customer_id"):
        params["customer"] = org["stripe_customer_id"]
    else:
        params["customer_creation"] = "always"
    session = s.checkout.Session.create(**params)
    return session.url


async def create_customer_portal_url(org: dict, return_url: str) -> str:
    s = _stripe()
    if not org.get("stripe_customer_id"):
        raise ValueError("Org has no Stripe customer yet")
    portal = s.billing_portal.Session.create(
        customer=org["stripe_customer_id"], return_url=return_url
    )
    return portal.url


def verify_webhook(payload: bytes, sig_header: str):
    """Returns the parsed Stripe event after signature verification, or raises."""
    s = _stripe()
    return s.Webhook.construct_event(
        payload, sig_header, config.STRIPE_WEBHOOK_SECRET
    )


async def handle_event(event: dict) -> None:
    """Idempotent dispatch — safe to call multiple times for the same event."""
    event_id = event["id"]
    event_type = event["type"]

    # Record (idempotent insert) and bail early if we've already processed it
    is_new = await db.record_stripe_event(event_id, event_type, event)
    if not is_new:
        if await db.stripe_event_seen(event_id):
            log.info("[stripe] event %s already processed, skipping", event_id)
            return

    log.info("[stripe] processing event %s type=%s", event_id, event_type)
    obj = event["data"]["object"]

    try:
        if event_type == "checkout.session.completed":
            await _on_checkout_completed(obj)
        elif event_type in ("customer.subscription.created", "customer.subscription.updated"):
            await _on_subscription_change(obj)
        elif event_type == "customer.subscription.deleted":
            await _on_subscription_deleted(obj)
        elif event_type == "invoice.payment_failed":
            await _on_payment_failed(obj)
        else:
            log.info("[stripe] unhandled event type %s", event_type)
    finally:
        await db.mark_stripe_event_processed(event_id)


def _ts_to_dt(ts: int | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


async def _on_checkout_completed(session: dict) -> None:
    """The user finished Stripe Checkout. Wire customer + subscription back to org."""
    org_id = (session.get("metadata") or {}).get("org_id") or session.get("client_reference_id")
    if not org_id:
        log.warning("[stripe] checkout.session.completed missing org_id")
        return
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")
    await db.update_org_plan(
        org_id, plan_tier="unlimited",
        stripe_customer_id=customer_id, stripe_subscription_id=subscription_id,
    )
    log.info("[stripe] org %s upgraded via checkout (sub=%s)", org_id, subscription_id)


async def _on_subscription_change(sub: dict) -> None:
    """Subscription created or updated — sync state."""
    sub_id = sub["id"]
    customer_id = sub.get("customer")
    org = await db.get_org_by_stripe_subscription(sub_id)
    if not org and customer_id:
        org = await db.get_org_by_stripe_customer(customer_id)
    if not org:
        log.warning("[stripe] subscription %s not linked to any org", sub_id)
        return
    status = sub.get("status")
    period_end = _ts_to_dt(sub.get("current_period_end"))
    # Active states keep the user on Unlimited; everything else falls back to free.
    if status in ("active", "trialing", "past_due"):
        new_tier = "unlimited"
    else:
        new_tier = "free"
    await db.update_org_plan(
        org["id"], plan_tier=new_tier,
        stripe_subscription_id=sub_id,
        subscription_active_until=period_end,
    )
    log.info("[stripe] org %s sub %s status=%s → plan=%s", org["slug"], sub_id, status, new_tier)


async def _on_subscription_deleted(sub: dict) -> None:
    sub_id = sub["id"]
    org = await db.get_org_by_stripe_subscription(sub_id)
    if not org:
        return
    await db.update_org_plan(
        org["id"], plan_tier="free",
        subscription_active_until=_ts_to_dt(sub.get("current_period_end")),
    )
    log.info("[stripe] org %s subscription cancelled — back to free", org["slug"])


async def _on_payment_failed(invoice: dict) -> None:
    customer_id = invoice.get("customer")
    if not customer_id:
        return
    org = await db.get_org_by_stripe_customer(customer_id)
    if org:
        log.warning("[stripe] payment failed for org=%s customer=%s", org["slug"], customer_id)
